/* a drat of SNMLM to predict bytes. It looks like it can get to 1.65 bits per byte
 * on a enwik8 file (see Hutter Prize).
 *
 * Copyright (C) 2024 Serguey Zefirov
 */

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void usage(char*pn) {
	fprintf(stderr, "usage: %s (c|d) infile outfile\n", pn);
	exit(1);
} /* usage */

#define	STATE_SIZE		(256)

static uint8_t state[STATE_SIZE];

static void init_state(void) {
	memset(state, 0, sizeof(state));
} /* init_state */

static void update_state(int ch) {
	int i, j, h, l;
	for(i=0;i<STATE_SIZE-1;i++) {
		state[i] = state[i+1];
	}
	state[i] = ch;
} /* update_state */

#define	NCHARS	(256)
static float probs[NCHARS];	// reserve EOF at end.
static float bias0;
static float bias[NCHARS];

typedef struct {
	uint64_t	mask[256 / 64];
} chars_mask;
typedef	struct {
	float*	x0;
	float*	x1;
	float	y;
	float	p;
	float	shifted_exp;
	int	ch;
} random_exp;
#define	NUM_FTS		(10000)	// about 39 full 256-char features
#define	NUM_MASKS	(1000)
#define	MAX_RWS		(3 << 28)	// float weights kept in hash map. 1GB.
#define	MAX_RCMS	(1 << 26)	// chars_mask's kept in hash map. 1GB
static random_exp fts[NUM_FTS];
static chars_mask* features_masks[NUM_MASKS];
static int num_random_exps;
static int num_features_masks;
static float max_logprob;
static float sumsum;
static float sums[NCHARS];
static float* ref;
static chars_mask* rcms;
static float prob_mul;

static void init_model(void) {
	int i;
	prob_mul = 0;
	bias0 = 0;
	for(i=0;i<NCHARS;i++) {
		bias[i] = 0;
	}
	size_t ref_size = sizeof(*ref) * MAX_RWS;
	size_t rcms_size = sizeof(*rcms) * MAX_RCMS;
	ref = malloc(ref_size);
	rcms = malloc(rcms_size);
	memset(ref, 0, ref_size);
	memset(rcms, 0, rcms_size);
} /* init_model */


static void add_ft(float* x0, float* x1, int ch, float prob) {
	float y = *x0 + *x1 + prob * prob_mul;
	if (max_logprob < y) {
		max_logprob = y;
	}
	assert(num_random_exps < NUM_FTS);
	fts[num_random_exps].x0 = x0;
	fts[num_random_exps].x1 = x1;
	fts[num_random_exps].y  = y;
	fts[num_random_exps].p = prob;
	fts[num_random_exps].ch = ch;
	num_random_exps ++;
} /* add_ft */
typedef struct {
	uint64_t	a, b;
} hash_state;
static const hash_state hash0 = {
	  0x08088405edb88320ULL
	, 0xaaaaaaaa55555555ULL
};
static hash_state hash_add(hash_state h, uint64_t x) {
	h.a ^= x;
	h.a *= 0x08088405ULL;
	h.a += h.a >> 32;
	h.b += h.a;
	h.b *= 0x7fffffedULL;
	h.b ^= h.b >> 32;
	h.b ^= h.b >> 48;
	h.b ^= x;
	return h;
} /* hash_add */
static int add_fts(int len, int skip) {
	int i;
	assert(len + skip < STATE_SIZE);
	hash_state h = hash_add(hash0, len * STATE_SIZE + skip);
	for (i=0;i<len;i++) {
		uint8_t c = state[STATE_SIZE - 1 - skip - i];
		//printf("++ %02x\n", c);
		h = hash_add(h, 65536 + c);
	}
	uint64_t x = h.a ^ h.b;
	uint64_t xx = h.a ^ h.b;
	x %= MAX_RCMS;
	xx %= MAX_RWS;
	//printf("feat %08lx: len %d skip %d\n", x, len, skip);
	chars_mask* p = &rcms[x];
	assert(num_features_masks < NUM_MASKS);
	features_masks[num_features_masks ++] = p;
	int add = 0;
	uint64_t ys[NCHARS];
	int chs[NCHARS];
	float xs[NCHARS];
	float maxx = -1e20;
	int nys = 0;
	for (i=0;i<256/64;i++) {
		uint64_t m = p->mask[i];
		if (!m) {
			continue;
		}
		int j;
		for (j=0;j<64;j++) {
			if (((m >> j) & 1) == 0) {
				continue;
			}
			int ch = i * 64 + j;
			//printf("  %08lx: add char %02x\n", x, ch);
			hash_state th = h;
			th = hash_add(th, ch);
			uint64_t y = th.a ^ th.b;
			y %= MAX_RWS;
			ys[nys] = y;
			xs[nys] = ref[y];
			maxx = maxx < xs[nys] ? xs[nys] : maxx;
			chs[nys] = ch;
			nys ++;
			add = 1;
		}
	}
	if (!nys) {
		return 0;
	}
	float ent = 0;
	float expsum = 0;
	for (i=0;i<nys;i++) {
		float z = xs[i] - maxx;
		float e = expf(z);
		expsum += e;
		ent -= e * z;
	}
	ent /= expsum;
	ent += logf(expsum);
	ent /= logf(NCHARS);
        for (i=0;i<nys;i++) {
		add_ft(&ref[x], &ref[ys[i]], chs[i], ent);
	}
	return nys > 1;
} /* add_fts */
static void infer(void) {
	int i;
	num_random_exps = 0;
	num_features_masks = 0;
	for(i=0;i<NCHARS;i++) {
		sums[i] = 0;
	}
	int j, k;
	max_logprob = -1e30;
	float mb = -1e10;
	float es[NCHARS];
	float esum = 0;
	for(i=0;i<NCHARS;i++) {
		if (mb < bias[i]) {
			mb = bias[i];
		}
	}
	float ent = 0;
	for(i=0;i<NCHARS;i++) {
		float z = bias[i] - mb;
		float e = expf(z);
		esum += e;
		ent -= e * z;
		es[i] = e;
	}
	ent /= esum;
	ent += logf(esum);
	ent /= logf(NCHARS);
	//printf("bias ent %g\n", ent);
	for(i=1;i<NCHARS;i++) {
		add_ft(&bias0, &bias[i], i, ent);
	}
	int skip;
	for (skip = 0; skip < 1; skip ++) {
		int nn = 16 - skip * 4;
		for (i=1;i<=nn;i++) {
			if (!add_fts(i, skip)) {
				break;
			}
		}
	}

	for(i=0;i<num_random_exps;i++) {
		int ch = fts[i].ch;
		float z = expf(fts[i].y - max_logprob);
		float mn = 1e-10;
		fts[i].shifted_exp = z < mn ? mn : z;
		sums[ch] += fts[i].shifted_exp;
	}
	sumsum = 0;
	for(i=0;i<NCHARS;i++) {
		sumsum += sums[i];
	}
	for(i=0;i<NCHARS;i++) {
		probs[i] = sums[i] / sumsum;
	}
} /* infer */

#define	START_LEARN_RATE	(1.0f)
#define	END_LEARN_RATE		(START_LEARN_RATE * 0.1f)
#define	LEARN_DECAY_ONE		(1000000000.0f)
static int64_t update_counter = 0;
static void update_model(int ch) {
	int i;
	float m = 1; //(LEARN_DECAY_ONE - update_counter) / LEARN_DECAY_ONE;
	float lr = START_LEARN_RATE * m + END_LEARN_RATE * (1 - m);
	update_counter ++;
	for (i=0;i<NCHARS;i++) {
		probs[i] = (i == ch ? 1.0f/sums[i] : 0) - 1/sumsum;	// (d/d(sums[i])) (log(probs[ch]))
	}
	for (i=0;i<num_random_exps;i++) {
		float* x0 = fts[i].x0;
		float* x1 = fts[i].x1;
		float d = probs[fts[i].ch] * fts[i].shifted_exp;
		*x0 += lr * d;
		*x1 += lr * d;
		prob_mul += lr * fts[i].p * d;
	}
	for(i=0;i<num_features_masks;i++) {
		chars_mask* p = features_masks[i];
		p->mask[ch/64] |= 1ULL << (ch % 64);
	}
} /* update_model */

static void predict(void) {
	infer();
} /* predict */

static void flush_encoder(void) {
	fprintf(stderr, "slush encoder\n");
	exit(1);
} /* flush_encoder */

#define	PART_SIZE	(100000)
double logsum = 0;
double part_logsum = 0;
uint64_t counter = 0;
static void encode(int ch) {
	float p = probs[ch];
	p = p < 1e-10 ? 1e-10 : p;
	double log2p = -log2(p);
	logsum += log2p;
	part_logsum += log2p;
	counter ++;
	if (counter > 0 && (counter % PART_SIZE) == 0) {
		printf("processed %9ld: total avg bits %8.5f, part avg bits %8.5f\n", counter, logsum/counter, part_logsum/PART_SIZE);
		part_logsum = 0;
	}
} /* encode */

int main(int argc, char** argv) {
	setvbuf( stdout, NULL, _IONBF, 0 );
	setvbuf( stderr, NULL, _IONBF, 0 );
	if (argc < 4) {
		usage(argv[0]);
	}
	int compress = 0;
	if (strcmp(argv[1], "c") == 0) {
		compress = 1;
	} else if (strcmp(argv[1], "d") == 0) {
		compress = 0;
	} else {
		usage(argv[0]);
	}
	FILE* inp = fopen(argv[2], "rb");
	if (!inp) {
		fprintf(stderr, "unable to open %s\n", argv[2]);
		return 1;
	}
	FILE* outp = fopen(argv[3], "wb");
	if (!outp) {
		fprintf(stderr, "unable to create %s\n", argv[3]);
		return 1;
	}
	init_state();
	init_model();
	printf("write or read uncompressed size here\n");
	while(1) {
		predict();
		int ch;
		if (compress) {
			ch = fgetc(inp);
			if (ch < 0) {
				flush_encoder();
				break;
			}
			encode(ch);
		} else {
			fprintf(stderr, "huh?\n");
			exit(1);
		}
		update_model(ch);
		update_state(ch);
	}
	fclose(inp);
	fclose(outp);
	return 0;
} /* main */

