/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, Carnegie Mellon University
 *
 * DCP PCI PMD - a DPDK poll-mode driver for DCP Ethernet devices.
 *
 * The device exposes two BARs:
 *   BAR0 - control / status registers (AXI-Lite, non-WC).
 *   BAR2 - data queue memory (write-combining).
 *
 * Device->Host (DPDK RX): the device DMA-writes DCP commands (InValue /
 * InRef / ReqBuf / ReqCred) into a consumer DMA ring allocated from huge-
 * pages.  The PMD polls this ring in rx_burst.
 *
 * Host->Device (DPDK TX): the PMD writes InRef commands (1 credit header)
 * into the device's RX data queue region in BAR2.  The referenced packet
 * data lives in a host-side DMA buffer; the device DMA-reads it.
 */

#include <stdint.h>
#include <string.h>
#include <inttypes.h>

#include <ethdev_driver.h>
#include <ethdev_pci.h>
#include <bus_pci_driver.h>
#include <rte_malloc.h>
#include <rte_memzone.h>
#include <rte_mbuf.h>
#include <rte_io.h>
#include <rte_log.h>

RTE_LOG_REGISTER_DEFAULT(dcp_logtype, NOTICE);
#define DCP_LOG(level, ...) \
	rte_log(RTE_LOG_ ## level, dcp_logtype, "DCP: " __VA_ARGS__)

/* ------------------------------------------------------------------ */
/*  DCP protocol constants (must match dcp/config.h & dcp_pkg.sv)     */
/* ------------------------------------------------------------------ */

#define DCP_VENDOR_ID           0x1234
#define DCP_DEVICE_ID           0x0002

#define DCP_CREDIT_SIZE         64
#define DCP_MAX_NUM_RX_QUEUES   2
#define DCP_MAX_NUM_TX_QUEUES   2

#define DCP_CORE_QUEUE_DEPTH    1024
#define DCP_CORE_MSS            16
#define DCP_CORE_MAX_REF_SIZE   512

/* Command types. */
#define DCP_CMD_EMPTY     0
#define DCP_CMD_IN_VALUE  1
#define DCP_CMD_IN_REF    2
#define DCP_CMD_REQ_BUF   3
#define DCP_CMD_REQ_CRED  4
#define DCP_CMD_RESP      5
#define DCP_CMD_CFG       7

/* Response types. */
#define DCP_RESP_NON_POSTED 1
#define DCP_RESP_BUF_REQ    2
#define DCP_RESP_CRED_REQ   3

/* Config types. */
#define DCP_CFG_RX 1
#define DCP_CFG_TX 2

/* BAR0 register offsets. */
#define DCP_REG_DMA_ENABLE        0x0000
#define DCP_REG_FUNNEL            0x1000
#define DCP_REG_PKTGEN_RST        0x1500
#define DCP_REG_PKTGEN_RUN        0x1504
#define DCP_REG_PKTGEN_NUM_LO     0x1508
#define DCP_REG_PKTGEN_NUM_HI     0x150c
#define DCP_REG_PKTGEN_SIZE       0x1510
#define DCP_REG_PKTGEN_RATE_NUM   0x1514
#define DCP_REG_PKTGEN_RATE_DEN   0x1518
#define DCP_REG_PKTGEN_QUEUES     0x151c
#define DCP_REG_PKTGEN_TIMESTAMP  0x1520
#define DCP_REG_PKTGEN_PRECISION  0x1524
#define DCP_REG_PKTGEN_HIST_SEL   0x1528
#define DCP_REG_PKTGEN_DONE       0x152c
#define DCP_REG_PKTGEN_RX_CYC_LO  0x1530
#define DCP_REG_PKTGEN_RX_CYC_HI  0x1534
#define DCP_REG_PKTGEN_RX_MSG_LO  0x1538
#define DCP_REG_PKTGEN_RX_MSG_HI  0x153c
#define DCP_REG_PKTGEN_TX_CYC_LO  0x1540
#define DCP_REG_PKTGEN_TX_CYC_HI  0x1544
#define DCP_REG_PKTGEN_TX_MSG_LO  0x1548
#define DCP_REG_PKTGEN_TX_MSG_HI  0x154c
#define DCP_REG_PKTGEN_CUM_LAT_LO 0x155c
#define DCP_REG_PKTGEN_CUM_LAT_HI 0x1560
#define DCP_REG_HW_MANAGED_BUFS   0x1700

#define DCP_MAX_INLINE_DATA \
	((uint32_t)(DCP_CORE_MSS * DCP_CREDIT_SIZE - 16))

/* ------------------------------------------------------------------ */
/*  Bit-packing helpers (match SV packed struct layout exactly)        */
/* ------------------------------------------------------------------ */

/*
 * DCP headers are defined as SystemVerilog packed structs with fields
 * laid out from LSB upward.  On the wire each 64-byte credit is a
 * little-endian byte stream: bit 0 → byte 0 bit 0.
 *
 * The first 256 bits (32 bytes) of every credit contain the header;
 * any inline payload data starts at byte 32.
 */
#define DCP_HEADER_BYTES 32

static inline uint64_t
dcp_unpack(const volatile uint8_t *buf, unsigned pos, unsigned width)
{
	/* Use __uint128_t to avoid UB when a 64-bit field crosses a byte
	 * boundary (shift by 64 is undefined for uint64_t on x86). */
	unsigned start_byte = pos / 8;
	unsigned start_bit  = pos % 8;
	unsigned nread = (start_bit + width + 7) / 8;
	__uint128_t acc = 0;

	for (unsigned i = 0; i < nread; i++)
		acc |= (__uint128_t)(uint8_t)buf[start_byte + i] << (i * 8);

	acc >>= start_bit;
	if (width < 64)
		return (uint64_t)(acc & ((1ULL << width) - 1));
	return (uint64_t)acc;
}

static inline void
dcp_pack(uint8_t *buf, unsigned pos, uint64_t val, unsigned width)
{
	if (width < 64)
		val &= (1ULL << width) - 1;

	unsigned start_byte = pos / 8;
	unsigned start_bit  = pos % 8;
	unsigned nwrite = (start_bit + width + 7) / 8;
	__uint128_t shifted = (__uint128_t)val << start_bit;

	for (unsigned i = 0; i < nwrite; i++) {
		buf[start_byte + i] |= (uint8_t)(shifted >> (i * 8));
	}
}

/* ---- Bit positions for each header type (from SV packed structs) ---- */

/* Common (all commands): cmd_type is bits [2:0]. */
#define BP_CMD_TYPE     0
#define BW_CMD_TYPE     3

/* in_value_cmd_header_t: cmd_type(3), posted(1), size(12), sender(64) */
#define BP_IV_POSTED    3
#define BW_IV_POSTED    1
#define BP_IV_SIZE      4
#define BW_IV_SIZE      12
#define BP_IV_SENDER    16
#define BW_IV_SENDER    64

/* in_ref_cmd_header_t: cmd_type(3), posted(1), size(16), sender(64), addr(64) */
#define BP_IR_POSTED    3
#define BW_IR_POSTED    1
#define BP_IR_SIZE      4
#define BW_IR_SIZE      16
#define BP_IR_SENDER    20
#define BW_IR_SENDER    64
#define BP_IR_ADDR      84
#define BW_IR_ADDR      64

/* req_buf_cmd_header_t: cmd_type(3), blocking(1), size(16), sender(64) */
#define BP_RB_BLOCKING  3
#define BP_RB_SIZE      4
#define BW_RB_SIZE      16
#define BP_RB_SENDER    20
#define BW_RB_SENDER    64

/* req_cred_cmd_header_t: cmd_type(3), blocking(1), num_credits(24), sender(64) */
#define BP_RC_BLOCKING  3
#define BP_RC_NCREDS    4
#define BW_RC_NCREDS    24
#define BP_RC_SENDER    28
#define BW_RC_SENDER    64

/* resp_buf_req_header_t: cmd_type(3), resp_type(3), success(1), size(16), addr(64) */
#define BP_RBR_RTYPE    3
#define BW_RBR_RTYPE    3
#define BP_RBR_SUCCESS  6
#define BW_RBR_SUCCESS  1
#define BP_RBR_SIZE     7
#define BW_RBR_SIZE     16
#define BP_RBR_ADDR     23
#define BW_RBR_ADDR     64

/* resp_cred_req_header_t: cmd_type(3), resp_type(3), num_credits(24) */
#define BP_RCR_RTYPE    3
#define BW_RCR_RTYPE    3
#define BP_RCR_NCREDS   6
#define BW_RCR_NCREDS   24

/* cfg_tx_cmd_header_t: cmd_type(3), cfg_type(3), tx_queue_id(1),
 *   initial_credits(24), cred_req_thresh(24), addr(64),
 *   sender_addr(64), queue_depth(25), mss(6), max_ref_msg_sz(11) */
#define BP_CTX_CFGTYPE  3
#define BW_CTX_CFGTYPE  3
#define BP_CTX_QID      6
#define BW_CTX_QID      1
#define BP_CTX_ICRED    7
#define BW_CTX_ICRED    24
#define BP_CTX_CRT      31
#define BW_CTX_CRT      24
#define BP_CTX_ADDR     55
#define BW_CTX_ADDR     64
#define BP_CTX_SADDR    119
#define BW_CTX_SADDR    64
#define BP_CTX_QDEPTH   183
#define BW_CTX_QDEPTH   25
#define BP_CTX_MSS      208
#define BW_CTX_MSS      6
#define BP_CTX_MAXREF   214
#define BW_CTX_MAXREF   11

/* cfg_rx_cmd_header_t: cmd_type(3), cfg_type(3),
 *   local_buf_addr(64), remote_buf_addr(64), remote_buf_len(64),
 *   init_credits(24) */
#define BP_CRX_CFGTYPE  3
#define BW_CRX_CFGTYPE  3
#define BP_CRX_LBUF     6
#define BW_CRX_LBUF     64
#define BP_CRX_RBUF     70
#define BW_CRX_RBUF     64
#define BP_CRX_RLEN     134
#define BW_CRX_RLEN     64
#define BP_CRX_ICRED    198
#define BW_CRX_ICRED    24

/* ------------------------------------------------------------------ */
/*  Adapter / queue structures                                        */
/* ------------------------------------------------------------------ */

#define DCP_MAX_BURST 64

struct dcp_rx_queue {
	const struct rte_memzone *mz;
	volatile uint8_t *ring;
	uint64_t          ring_iova;
	uint32_t          head;
	uint32_t          size;
	uint32_t          freed_creds;
	struct rte_mempool *mb_pool;
	uint16_t          port_id;
	uint16_t          queue_id;
	void             *adapter;
};

struct dcp_tx_queue {
	volatile uint8_t *ring;
	uint32_t          tail;
	uint32_t          size;
	uint32_t          credits;
	uint16_t          port_id;
	uint16_t          queue_id;
	void             *adapter;
};

struct dcp_adapter {
	volatile uint32_t *bar0;
	volatile uint8_t  *bar2;
	uint64_t           bar2_phys;
	uint32_t           queue_depth;
	uint16_t           mss;

	const struct rte_memzone *alloc_mz;
	uint8_t   *alloc_vaddr;
	uint64_t   alloc_iova;
	uint32_t   alloc_size;
	uint32_t   alloc_tail;
	uint32_t   alloc_head;

	/* TX DMA buffer: host memory where packet data is placed
	 * for InRef sends.  The device DMA-reads from here. */
	const struct rte_memzone *tx_buf_mz;
	uint8_t   *tx_buf_vaddr;
	uint64_t   tx_buf_iova;
	uint32_t   tx_buf_size;
	uint32_t   tx_buf_tail;

	uint16_t   nb_rx_queues;
	uint16_t   nb_tx_queues;
	uint32_t   cfg_tail;

	struct dcp_rx_queue *rxqs[DCP_MAX_NUM_RX_QUEUES];
	struct dcp_tx_queue *txqs[DCP_MAX_NUM_TX_QUEUES];
};

/* ------------------------------------------------------------------ */
/*  BAR helpers                                                       */
/* ------------------------------------------------------------------ */

static inline void
dcp_write32(volatile uint32_t *bar0, uint32_t off, uint32_t val)
{
	rte_write32(val, (volatile void *)((uintptr_t)bar0 + off));
}

static inline uint32_t
dcp_read32(volatile uint32_t *bar0, uint32_t off)
{
	return rte_read32((const volatile void *)((uintptr_t)bar0 + off));
}

/* ------------------------------------------------------------------ */
/*  Allocator (simple bump ring for InRef buffers)                    */
/* ------------------------------------------------------------------ */

static uint64_t
dcp_alloc_buf(struct dcp_adapter *a, uint32_t size_bytes)
{
	uint32_t creds = (size_bytes + DCP_CREDIT_SIZE - 1) / DCP_CREDIT_SIZE;
	uint32_t ring_creds = a->alloc_size / DCP_CREDIT_SIZE;
	uint32_t mask = ring_creds - 1;
	/* Use monotonic head/tail: avail = capacity - in_flight. */
	uint32_t used  = a->alloc_tail - a->alloc_head;
	uint32_t avail = ring_creds - used;

	if (creds > avail)
		return 0;

	uint32_t pos = a->alloc_tail & mask;
	a->alloc_tail += creds;
	return a->alloc_iova + (uint64_t)pos * DCP_CREDIT_SIZE;
}

static void
dcp_free_alloc(struct dcp_adapter *a, uint32_t size_bytes)
{
	uint32_t creds = (size_bytes + DCP_CREDIT_SIZE - 1) / DCP_CREDIT_SIZE;
	a->alloc_head += creds;
}

/* ------------------------------------------------------------------ */
/*  Config-queue helpers (write to BAR2 queue 0)                      */
/* ------------------------------------------------------------------ */

static void
dcp_send_cfg_raw(struct dcp_adapter *a, const void *hdr, size_t hdr_sz)
{
	volatile uint8_t *base = a->bar2; /* queue 0 at offset 0 */
	uint32_t pos = a->cfg_tail % a->queue_depth;

	uint8_t buf[DCP_CREDIT_SIZE];
	memset(buf, 0, sizeof(buf));
	memcpy(buf, hdr, hdr_sz);

	/* Hex dump first 32 bytes for debugging. */
	DCP_LOG(NOTICE, "cfg_raw[%u] @BAR2+0x%lx hdr_sz=%zu: "
		"%02x%02x%02x%02x %02x%02x%02x%02x "
		"%02x%02x%02x%02x %02x%02x%02x%02x "
		"%02x%02x%02x%02x %02x%02x%02x%02x "
		"%02x%02x%02x%02x %02x%02x%02x%02x\n",
		a->cfg_tail,
		(unsigned long)((uint64_t)pos * DCP_CREDIT_SIZE),
		hdr_sz,
		buf[0], buf[1], buf[2], buf[3],
		buf[4], buf[5], buf[6], buf[7],
		buf[8], buf[9], buf[10], buf[11],
		buf[12], buf[13], buf[14], buf[15],
		buf[16], buf[17], buf[18], buf[19],
		buf[20], buf[21], buf[22], buf[23],
		buf[24], buf[25], buf[26], buf[27],
		buf[28], buf[29], buf[30], buf[31]);

	rte_memcpy((void *)(uintptr_t)(base + (uint64_t)pos * DCP_CREDIT_SIZE),
		    buf, DCP_CREDIT_SIZE);
	rte_wmb();
	a->cfg_tail++;
}

static void
dcp_send_cfg_tx(struct dcp_adapter *a, uint8_t tx_qid,
		uint64_t cons_ring_iova, uint32_t depth,
		uint16_t mss, uint16_t max_ref,
		uint64_t sender_phys, uint32_t init_creds)
{
	uint8_t buf[DCP_CREDIT_SIZE];
	memset(buf, 0, sizeof(buf));
	dcp_pack(buf, BP_CMD_TYPE,   DCP_CMD_CFG,  BW_CMD_TYPE);
	dcp_pack(buf, BP_CTX_CFGTYPE, DCP_CFG_TX,  BW_CTX_CFGTYPE);
	dcp_pack(buf, BP_CTX_QID,   tx_qid,        BW_CTX_QID);
	dcp_pack(buf, BP_CTX_ICRED, init_creds,     BW_CTX_ICRED);
	dcp_pack(buf, BP_CTX_CRT,   init_creds / 2, BW_CTX_CRT);
	dcp_pack(buf, BP_CTX_ADDR,  cons_ring_iova, BW_CTX_ADDR);
	dcp_pack(buf, BP_CTX_SADDR, sender_phys,    BW_CTX_SADDR);
	dcp_pack(buf, BP_CTX_QDEPTH, depth,         BW_CTX_QDEPTH);
	dcp_pack(buf, BP_CTX_MSS,   mss,            BW_CTX_MSS);
	dcp_pack(buf, BP_CTX_MAXREF, max_ref,       BW_CTX_MAXREF);
	dcp_send_cfg_raw(a, buf, DCP_CREDIT_SIZE);
}

static void
dcp_send_cfg_rx(struct dcp_adapter *a, uint64_t local_buf,
		uint64_t remote_buf, uint64_t remote_len,
		uint32_t init_creds)
{
	uint8_t buf[DCP_CREDIT_SIZE];
	memset(buf, 0, sizeof(buf));
	dcp_pack(buf, BP_CMD_TYPE,   DCP_CMD_CFG, BW_CMD_TYPE);
	dcp_pack(buf, BP_CRX_CFGTYPE, DCP_CFG_RX, BW_CRX_CFGTYPE);
	dcp_pack(buf, BP_CRX_LBUF,  local_buf,   BW_CRX_LBUF);
	dcp_pack(buf, BP_CRX_RBUF,  remote_buf,  BW_CRX_RBUF);
	dcp_pack(buf, BP_CRX_RLEN,  remote_len,  BW_CRX_RLEN);
	dcp_pack(buf, BP_CRX_ICRED, init_creds,  BW_CRX_ICRED);
	dcp_send_cfg_raw(a, buf, DCP_CREDIT_SIZE);
}

/* ------------------------------------------------------------------ */
/*  RX burst                                                          */
/* ------------------------------------------------------------------ */

static inline void
dcp_write_resp_to_bar2(struct dcp_adapter *a, uint64_t sender_phys,
		       const void *resp, size_t sz)
{
	uint64_t off = sender_phys - a->bar2_phys;
	volatile uint8_t *dst = a->bar2 + off;
	uint8_t buf[DCP_CREDIT_SIZE];
	memset(buf, 0, sizeof(buf));
	memcpy(buf, resp, sz);
	rte_memcpy((void *)(uintptr_t)dst, buf, DCP_CREDIT_SIZE);
	rte_wmb();
}

static uint16_t
eth_dcp_rx(void *rxq_ptr, struct rte_mbuf **pkts, uint16_t nb_pkts)
{
	struct dcp_rx_queue *rxq = rxq_ptr;
	struct dcp_adapter *a = rxq->adapter;
	uint16_t nb_rx = 0;
	uint32_t processed = 0;
	uint32_t mask = rxq->size - 1;

	/* One-time dump of first slot for debugging. */
	static int rx_dump_done[2] = {0, 0};
	if (rxq->queue_id < 2 && !rx_dump_done[rxq->queue_id]) {
		rx_dump_done[rxq->queue_id] = 1;
		volatile uint8_t *s0 = rxq->ring;
		DCP_LOG(NOTICE, "RX q%u ring=%p iova=0x%lx head=%u first8: "
			"%02x%02x%02x%02x %02x%02x%02x%02x\n",
			rxq->queue_id, rxq->ring,
			(unsigned long)rxq->ring_iova,
			rxq->head,
			s0[0], s0[1], s0[2], s0[3],
			s0[4], s0[5], s0[6], s0[7]);
	}

	while (nb_rx < nb_pkts) {
		volatile uint8_t *slot = rxq->ring +
			(uint64_t)(rxq->head & mask) * DCP_CREDIT_SIZE;
		uint8_t ct = (uint8_t)dcp_unpack(slot, BP_CMD_TYPE, BW_CMD_TYPE);

		if (ct == DCP_CMD_EMPTY)
			break;

		if (ct == DCP_CMD_IN_VALUE) {
			uint32_t sz = (uint32_t)dcp_unpack(slot,
					BP_IV_SIZE, BW_IV_SIZE);
			uint32_t ncreds =
				(DCP_HEADER_BYTES + sz +
				 DCP_CREDIT_SIZE - 1) / DCP_CREDIT_SIZE;

			struct rte_mbuf *m = rte_pktmbuf_alloc(rxq->mb_pool);
			if (unlikely(!m))
				break;
			const uint8_t *src = (const uint8_t *)(uintptr_t)slot +
				DCP_HEADER_BYTES;
			memcpy(rte_pktmbuf_mtod(m, void *), src, sz);
			m->data_len = sz;
			m->pkt_len  = sz;
			m->port     = rxq->port_id;
			pkts[nb_rx++] = m;

			for (uint32_t c = 0; c < ncreds; c++) {
				volatile uint64_t *s = (volatile uint64_t *)
					(rxq->ring +
					 (uint64_t)((rxq->head + c) & mask) *
					 DCP_CREDIT_SIZE);
				*s = 0;
			}
			rxq->head += ncreds;
			processed += ncreds;

		} else if (ct == DCP_CMD_IN_REF) {
			uint32_t sz   = (uint32_t)dcp_unpack(slot,
					BP_IR_SIZE, BW_IR_SIZE);
			uint64_t addr = dcp_unpack(slot,
					BP_IR_ADDR, BW_IR_ADDR);

			struct rte_mbuf *m = rte_pktmbuf_alloc(rxq->mb_pool);
			if (unlikely(!m))
				break;

			int valid = 0;
			if (addr >= a->bar2_phys &&
			    addr < a->bar2_phys + 0x1000000) {
				/* HW-managed bufs: data in BAR2 spill. */
				uint64_t boff = addr - a->bar2_phys;
				rte_memcpy(rte_pktmbuf_mtod(m, void *),
				       (void *)(uintptr_t)(a->bar2 + boff),
				       sz);
				valid = 1;
			} else if (addr >= a->alloc_iova &&
				   addr < a->alloc_iova + a->alloc_size) {
				/* Non-HW-managed: data in host alloc. */
				uint64_t roff = addr - a->alloc_iova;
				memcpy(rte_pktmbuf_mtod(m, void *),
				       a->alloc_vaddr + roff, sz);
				dcp_free_alloc(a, sz);
				valid = 1;
			}
			if (!valid) {
				DCP_LOG(ERR, "InRef bad addr=0x%" PRIx64
					" sz=%u\n", addr, sz);
				rte_pktmbuf_free(m);
				*(volatile uint64_t *)(uintptr_t)slot = 0;
				rxq->head += 1;
				processed += 1;
				continue;
			}
			m->data_len = sz;
			m->pkt_len  = sz;
			m->port     = rxq->port_id;
			pkts[nb_rx++] = m;

			*(volatile uint64_t *)(uintptr_t)slot = 0;
			rxq->head += 1;
			processed += 1;

		} else if (ct == DCP_CMD_REQ_BUF) {
			uint32_t sz     = (uint32_t)dcp_unpack(slot,
					BP_RB_SIZE, BW_RB_SIZE);
			uint64_t sender = dcp_unpack(slot,
					BP_RB_SENDER, BW_RB_SENDER);
			uint64_t buf_addr = dcp_alloc_buf(a, sz);

			uint8_t resp[DCP_CREDIT_SIZE];
			memset(resp, 0, sizeof(resp));
			dcp_pack(resp, BP_CMD_TYPE, DCP_CMD_RESP, BW_CMD_TYPE);
			dcp_pack(resp, BP_RBR_RTYPE, DCP_RESP_BUF_REQ,
				 BW_RBR_RTYPE);
			dcp_pack(resp, BP_RBR_SUCCESS,
				 (buf_addr != 0) ? 1 : 0, BW_RBR_SUCCESS);
			dcp_pack(resp, BP_RBR_SIZE, sz, BW_RBR_SIZE);
			dcp_pack(resp, BP_RBR_ADDR, buf_addr, BW_RBR_ADDR);
			dcp_write_resp_to_bar2(a, sender,
					       resp, sizeof(resp));

			*(volatile uint64_t *)(uintptr_t)slot = 0;
			rxq->head += 1;
			processed += 1;

		} else if (ct == DCP_CMD_REQ_CRED) {
			uint64_t sender = dcp_unpack(slot,
					BP_RC_SENDER, BW_RC_SENDER);
			uint32_t avail  = processed + rxq->freed_creds;
			rxq->freed_creds = 0;
			processed = 0;

			uint8_t resp[DCP_CREDIT_SIZE];
			memset(resp, 0, sizeof(resp));
			dcp_pack(resp, BP_CMD_TYPE, DCP_CMD_RESP, BW_CMD_TYPE);
			dcp_pack(resp, BP_RCR_RTYPE, DCP_RESP_CRED_REQ,
				 BW_RCR_RTYPE);
			dcp_pack(resp, BP_RCR_NCREDS, avail, BW_RCR_NCREDS);
			dcp_write_resp_to_bar2(a, sender,
					       resp, sizeof(resp));

			*(volatile uint64_t *)(uintptr_t)slot = 0;
			rxq->head += 1;
			processed += 1;

		} else {
			*(volatile uint64_t *)(uintptr_t)slot = 0;
			rxq->head += 1;
			processed += 1;
		}
	}

	rxq->freed_creds += processed;
	return nb_rx;
}

/* ------------------------------------------------------------------ */
/*  TX burst                                                          */
/* ------------------------------------------------------------------ */

static uint16_t
eth_dcp_tx(void *txq_ptr, struct rte_mbuf **pkts, uint16_t nb_pkts)
{
	struct dcp_tx_queue *txq = txq_ptr;
	struct dcp_adapter *a = txq->adapter;
	uint16_t nb_tx = 0;
	uint32_t mask = txq->size - 1;

	for (uint16_t i = 0; i < nb_pkts; i++) {
		struct rte_mbuf *m = pkts[i];
		uint32_t pkt_len = m->pkt_len;

		/* Each InRef command costs exactly 1 credit. */
		if (txq->credits < 1)
			break;

		/* Copy packet data to the TX DMA buffer so the device
		 * can DMA-read it after we write the InRef header. */
		uint32_t align_len = RTE_ALIGN_CEIL(pkt_len, DCP_CREDIT_SIZE);
		if (a->tx_buf_tail + align_len > a->tx_buf_size)
			break; /* out of TX buffer space */

		memcpy(a->tx_buf_vaddr + a->tx_buf_tail,
		       rte_pktmbuf_mtod(m, void *), pkt_len);
		uint64_t data_iova = a->tx_buf_iova + a->tx_buf_tail;
		a->tx_buf_tail += align_len;

		/* Build InRef header (fits in 1 credit). */
		uint8_t buf[DCP_CREDIT_SIZE];
		memset(buf, 0, sizeof(buf));
		dcp_pack(buf, BP_CMD_TYPE, DCP_CMD_IN_REF, BW_CMD_TYPE);
		dcp_pack(buf, BP_IR_POSTED, 1, BW_IR_POSTED);
		dcp_pack(buf, BP_IR_SIZE, pkt_len, BW_IR_SIZE);
		dcp_pack(buf, BP_IR_SENDER, 0, BW_IR_SENDER);
		dcp_pack(buf, BP_IR_ADDR, data_iova, BW_IR_ADDR);

		/* Compiler barrier: ensure the DMA data buffer stores
		 * are emitted before the BAR2 command write. */
		rte_compiler_barrier();

		/* Write 1 credit to BAR2 (write-combined region). */
		uint32_t pos = txq->tail & mask;
		volatile uint8_t *dst = txq->ring +
			(uint64_t)pos * DCP_CREDIT_SIZE;
		rte_memcpy((void *)(uintptr_t)dst, buf, DCP_CREDIT_SIZE);

		/* Flush the write-combining buffer so the device sees
		 * the InRef command. */
		rte_wmb();

		txq->tail    += 1;
		txq->credits -= 1;
		nb_tx++;

		rte_pktmbuf_free(m);
	}

	return nb_tx;
}

/* ------------------------------------------------------------------ */
/*  eth_dev_ops implementation                                        */
/* ------------------------------------------------------------------ */

static int
eth_dcp_dev_configure(struct rte_eth_dev *dev __rte_unused)
{
	return 0;
}

static int
eth_dcp_rx_queue_setup(struct rte_eth_dev *dev, uint16_t qid,
		       uint16_t nb_desc __rte_unused,
		       unsigned int socket_id __rte_unused,
		       const struct rte_eth_rxconf *conf __rte_unused,
		       struct rte_mempool *mb_pool)
{
	struct dcp_adapter *a = dev->data->dev_private;
	struct dcp_rx_queue *rxq;

	rxq = rte_zmalloc(NULL, sizeof(*rxq), RTE_CACHE_LINE_SIZE);
	if (!rxq)
		return -ENOMEM;

	uint32_t ring_bytes = (uint32_t)a->queue_depth * DCP_CREDIT_SIZE;
	char name[RTE_MEMZONE_NAMESIZE];
	snprintf(name, sizeof(name), "dcp_rx_%u_%u",
		 dev->data->port_id, qid);

	rxq->mz = rte_memzone_reserve_aligned(name, ring_bytes,
					       SOCKET_ID_ANY,
					       0,
					       RTE_CACHE_LINE_SIZE);
	if (!rxq->mz) {
		rte_free(rxq);
		return -ENOMEM;
	}
	memset(rxq->mz->addr, 0, ring_bytes);
	rxq->ring        = rxq->mz->addr;
	rxq->ring_iova   = rxq->mz->iova;
	rxq->size        = a->queue_depth;
	rxq->head        = 0;
	rxq->freed_creds = 0;
	rxq->mb_pool     = mb_pool;
	rxq->port_id     = dev->data->port_id;
	rxq->queue_id    = qid;
	rxq->adapter     = a;

	a->rxqs[qid] = rxq;
	dev->data->rx_queues[qid] = rxq;
	return 0;
}

static int
eth_dcp_tx_queue_setup(struct rte_eth_dev *dev, uint16_t qid,
		       uint16_t nb_desc __rte_unused,
		       unsigned int socket_id __rte_unused,
		       const struct rte_eth_txconf *conf __rte_unused)
{
	struct dcp_adapter *a = dev->data->dev_private;
	struct dcp_tx_queue *txq;

	txq = rte_zmalloc(NULL, sizeof(*txq), RTE_CACHE_LINE_SIZE);
	if (!txq)
		return -ENOMEM;

	/* Device RX data queue = qid + 1 (queue 0 is cfg). */
	uint64_t off = (uint64_t)(qid + 1) * a->queue_depth * DCP_CREDIT_SIZE;
	txq->ring     = a->bar2 + off;
	txq->tail     = 0;
	txq->size     = a->queue_depth;
	txq->credits  = a->queue_depth - 1;
	txq->port_id  = dev->data->port_id;
	txq->queue_id = qid;
	txq->adapter  = a;

	a->txqs[qid] = txq;
	dev->data->tx_queues[qid] = txq;
	return 0;
}

static void
eth_dcp_rx_queue_release(struct rte_eth_dev *dev, uint16_t qid)
{
	struct dcp_rx_queue *rxq = dev->data->rx_queues[qid];
	if (rxq) {
		if (rxq->mz)
			rte_memzone_free(rxq->mz);
		rte_free(rxq);
		dev->data->rx_queues[qid] = NULL;
	}
}

static void
eth_dcp_tx_queue_release(struct rte_eth_dev *dev, uint16_t qid)
{
	struct dcp_tx_queue *txq = dev->data->tx_queues[qid];
	if (txq) {
		rte_free(txq);
		dev->data->tx_queues[qid] = NULL;
	}
}

static int
eth_dcp_dev_start(struct rte_eth_dev *dev)
{
	struct dcp_adapter *a = dev->data->dev_private;
	uint16_t nrx = dev->data->nb_rx_queues;

	DCP_LOG(NOTICE, "dev_start: nrx=%u ntx=%u\n",
		nrx, dev->data->nb_tx_queues);

	/* ---- Phase 1: Reset device to clean state. ---- */

	/* Stop and reset pktgen. */
	dcp_write32(a->bar0, DCP_REG_PKTGEN_RUN, 0);
	dcp_write32(a->bar0, DCP_REG_PKTGEN_RST, 1);
	dcp_write32(a->bar0, DCP_REG_PKTGEN_RST, 0);

	/* Unbind all TX queues (send zero CfgTx for each). */
	for (uint16_t q = 0; q < DCP_MAX_NUM_TX_QUEUES; q++) {
		uint8_t buf[DCP_CREDIT_SIZE];
		memset(buf, 0, sizeof(buf));
		dcp_pack(buf, BP_CMD_TYPE, DCP_CMD_CFG, BW_CMD_TYPE);
		dcp_pack(buf, BP_CTX_CFGTYPE, DCP_CFG_TX, BW_CTX_CFGTYPE);
		dcp_pack(buf, BP_CTX_QID, q, BW_CTX_QID);
		dcp_send_cfg_raw(a, buf, DCP_CREDIT_SIZE);
	}

	/* Disable funnel traffic (not needed for single-device). */
	dcp_write32(a->bar0, DCP_REG_FUNNEL, 0);

	/* Send CfgRx with zero credits. */
	{
		uint8_t buf[DCP_CREDIT_SIZE];
		memset(buf, 0, sizeof(buf));
		dcp_pack(buf, BP_CMD_TYPE, DCP_CMD_CFG, BW_CMD_TYPE);
		dcp_pack(buf, BP_CRX_CFGTYPE, DCP_CFG_RX, BW_CRX_CFGTYPE);
		/* init_credits = 0 → no pack needed (buffer is zeroed). */
		dcp_send_cfg_raw(a, buf, DCP_CREDIT_SIZE);
	}

	DCP_LOG(NOTICE, "device reset complete\n");

	/* ---- Phase 2: Enable DMA engine and HW-managed buffers. ---- */
	dcp_write32(a->bar0, DCP_REG_DMA_ENABLE, 1);
	dcp_write32(a->bar0, DCP_REG_HW_MANAGED_BUFS, 1);

	/* ---- Phase 3: Allocate memory regions. ---- */

	/* Allocate remote-buffer region for InRef from device. */
	if (!a->alloc_mz) {
		char name[RTE_MEMZONE_NAMESIZE];
		uint32_t alloc_bytes = 2u * 1024 * 1024;
		snprintf(name, sizeof(name), "dcp_alloc_%u",
			 dev->data->port_id);
		a->alloc_mz = rte_memzone_reserve_aligned(name,
			alloc_bytes, SOCKET_ID_ANY,
			0, RTE_CACHE_LINE_SIZE);
		if (!a->alloc_mz) {
			DCP_LOG(ERR, "alloc_mz reserve failed\n");
			return -ENOMEM;
		}
		a->alloc_vaddr = a->alloc_mz->addr;
		a->alloc_iova  = a->alloc_mz->iova;
		a->alloc_size  = alloc_bytes;
		a->alloc_tail  = 0;
		a->alloc_head  = 0;
	}

	/* Allocate TX DMA buffer for InRef sends (host -> device). */
	if (!a->tx_buf_mz) {
		char name[RTE_MEMZONE_NAMESIZE];
		uint32_t tx_bytes = 2u * 1024 * 1024;
		snprintf(name, sizeof(name), "dcp_txbuf_%u",
			 dev->data->port_id);
		a->tx_buf_mz = rte_memzone_reserve_aligned(name,
			tx_bytes, SOCKET_ID_ANY,
			0, RTE_CACHE_LINE_SIZE);
		if (!a->tx_buf_mz) {
			DCP_LOG(ERR, "tx_buf_mz reserve failed\n");
			return -ENOMEM;
		}
		a->tx_buf_vaddr = a->tx_buf_mz->addr;
		a->tx_buf_iova  = a->tx_buf_mz->iova;
		a->tx_buf_size  = tx_bytes;
		a->tx_buf_tail  = 0;
	}

	DCP_LOG(NOTICE, "alloc_mz iova=0x%" PRIx64 " tx_buf iova=0x%" PRIx64 "\n",
		a->alloc_iova, a->tx_buf_iova);

	/* ---- Phase 4: Configure queues (CfgTx then CfgRx). ---- */

	/* Device-buffer addr in BAR2 spill region (after ALL queues:
	 * NUM_RX consumer queues + NUM_TX response queues). */
	uint64_t dev_buf_phys = a->bar2_phys +
		(uint64_t)(DCP_MAX_NUM_RX_QUEUES + DCP_MAX_NUM_TX_QUEUES) *
		a->queue_depth * DCP_CREDIT_SIZE;

	/* CfgTx for each device TX queue (= DPDK RX queue). */
	for (uint16_t q = 0; q < nrx; q++) {
		struct dcp_rx_queue *rxq = a->rxqs[q];
		if (!rxq)
			continue;

		uint64_t sender_phys = a->bar2_phys +
			(uint64_t)(DCP_MAX_NUM_RX_QUEUES + q) *
			a->queue_depth * DCP_CREDIT_SIZE;

		dcp_send_cfg_tx(a, (uint8_t)q, rxq->ring_iova,
				rxq->size, a->mss,
				DCP_CORE_MAX_REF_SIZE,
				sender_phys, rxq->size - 1);

		DCP_LOG(NOTICE, "CfgTx q=%u ring_iova=0x%" PRIx64
			" sender=0x%" PRIx64 " depth=%u mss=%u\n",
			q, rxq->ring_iova, sender_phys,
			rxq->size, a->mss);
	}

	/* CfgRx: tell device about our allocator region. */
	dcp_send_cfg_rx(a, dev_buf_phys, a->alloc_iova,
			a->alloc_size, a->queue_depth - 1);

	DCP_LOG(NOTICE, "CfgRx dev_buf=0x%" PRIx64 " remote=0x%" PRIx64
		" len=0x%x credits=%u\n",
		dev_buf_phys, a->alloc_iova, a->alloc_size,
		a->queue_depth - 1);

	dev->data->dev_link.link_status = RTE_ETH_LINK_UP;
	dev->data->dev_link.link_speed  = RTE_ETH_SPEED_NUM_100G;
	dev->data->dev_link.link_duplex = RTE_ETH_LINK_FULL_DUPLEX;
	return 0;
}

static int
eth_dcp_dev_stop(struct rte_eth_dev *dev)
{
	struct dcp_adapter *a = dev->data->dev_private;
	dcp_write32(a->bar0, DCP_REG_DMA_ENABLE, 0);
	dev->data->dev_link.link_status = RTE_ETH_LINK_DOWN;
	return 0;
}

static int
eth_dcp_dev_close(struct rte_eth_dev *dev)
{
	struct dcp_adapter *a = dev->data->dev_private;
	if (a->alloc_mz) {
		rte_memzone_free(a->alloc_mz);
		a->alloc_mz = NULL;
	}
	if (a->tx_buf_mz) {
		rte_memzone_free(a->tx_buf_mz);
		a->tx_buf_mz = NULL;
	}
	return 0;
}

static int
eth_dcp_dev_info(struct rte_eth_dev *dev __rte_unused,
		 struct rte_eth_dev_info *info)
{
	info->max_rx_queues = DCP_MAX_NUM_RX_QUEUES;
	/* Only (NUM_RX - 1) consumer data queues for TX (queue 0 is config). */
	info->max_tx_queues = DCP_MAX_NUM_RX_QUEUES - 1;
	info->min_rx_bufsize = DCP_CREDIT_SIZE;
	info->max_rx_pktlen  = RTE_ETHER_MAX_LEN;
	return 0;
}

static int
eth_dcp_link_update(struct rte_eth_dev *dev __rte_unused,
		    int wait __rte_unused)
{
	return 0;
}

static int
eth_dcp_promiscuous_enable(struct rte_eth_dev *dev __rte_unused)
{
	return 0;
}

static const struct eth_dev_ops dcp_eth_dev_ops = {
	.dev_configure      = eth_dcp_dev_configure,
	.dev_start          = eth_dcp_dev_start,
	.dev_stop           = eth_dcp_dev_stop,
	.dev_close          = eth_dcp_dev_close,
	.dev_infos_get      = eth_dcp_dev_info,
	.rx_queue_setup     = eth_dcp_rx_queue_setup,
	.tx_queue_setup     = eth_dcp_tx_queue_setup,
	.rx_queue_release   = eth_dcp_rx_queue_release,
	.tx_queue_release   = eth_dcp_tx_queue_release,
	.link_update        = eth_dcp_link_update,
	.promiscuous_enable = eth_dcp_promiscuous_enable,
};



/* ------------------------------------------------------------------ */
/*  PCI probe / remove                                                */
/* ------------------------------------------------------------------ */

static int
eth_dcp_dev_init(struct rte_eth_dev *dev)
{
	struct rte_pci_device *pci = RTE_ETH_DEV_TO_PCI(dev);
	struct dcp_adapter *a = dev->data->dev_private;

	dev->dev_ops      = &dcp_eth_dev_ops;
	dev->rx_pkt_burst = &eth_dcp_rx;
	dev->tx_pkt_burst = &eth_dcp_tx;

	/* BARs are mapped by DPDK via uio_pci_generic (RTE_PCI_DRV_NEED_MAPPING). */
	a->bar0      = (volatile uint32_t *)pci->mem_resource[0].addr;
	a->bar2      = (volatile uint8_t  *)pci->mem_resource[2].addr;
	a->bar2_phys = pci->mem_resource[2].phys_addr;

	if (!a->bar0 || !a->bar2) {
		DCP_LOG(ERR, "BAR mapping failed (BAR0=%p BAR2=%p)\n",
			pci->mem_resource[0].addr, pci->mem_resource[2].addr);
		return -ENODEV;
	}

	a->queue_depth = DCP_CORE_QUEUE_DEPTH;
	a->mss         = 1;  /* SW always sends 1-credit InRef commands. */
	a->cfg_tail    = 0;

	/* Allocate MAC address storage (not done by generic probe). */
	dev->data->mac_addrs = rte_zmalloc("dcp_mac",
		sizeof(struct rte_ether_addr), 0);
	if (!dev->data->mac_addrs)
		return -ENOMEM;

	struct rte_ether_addr mac = {
		.addr_bytes = { 0x00, 0x11, 0x22, 0x33, 0x44, 0x55 }
	};
	rte_ether_addr_copy(&mac, dev->data->mac_addrs);

	DCP_LOG(NOTICE, "DCP device initialized (BAR0=%p BAR2=%p phys=0x%"
		PRIx64 ")\n",
		(const volatile void *)a->bar0,
		(const volatile void *)a->bar2, a->bar2_phys);
	return 0;
}

static int
eth_dcp_dev_uninit(struct rte_eth_dev *dev __rte_unused)
{
	return 0;
}

static int
eth_dcp_pci_probe(struct rte_pci_driver *drv __rte_unused,
		  struct rte_pci_device *pci_dev)
{
	return rte_eth_dev_pci_generic_probe(pci_dev,
		sizeof(struct dcp_adapter), eth_dcp_dev_init);
}

static int
eth_dcp_pci_remove(struct rte_pci_device *pci_dev)
{
	return rte_eth_dev_pci_generic_remove(pci_dev, eth_dcp_dev_uninit);
}

/* ------------------------------------------------------------------ */
/*  PCI driver registration                                           */
/* ------------------------------------------------------------------ */

static const struct rte_pci_id pci_id_dcp_map[] = {
	{ RTE_PCI_DEVICE(DCP_VENDOR_ID, DCP_DEVICE_ID) },
	{ .vendor_id = 0 },
};

static struct rte_pci_driver rte_dcp_pmd = {
	.id_table  = pci_id_dcp_map,
	.drv_flags = RTE_PCI_DRV_NEED_MAPPING,
	.probe     = eth_dcp_pci_probe,
	.remove    = eth_dcp_pci_remove,
};

RTE_PMD_REGISTER_PCI(net_dcp, rte_dcp_pmd);
RTE_PMD_REGISTER_PCI_TABLE(net_dcp, pci_id_dcp_map);
