# GeneMamba Algorithm (Mathematical Formulation with Dimensions)

This note formalizes the algorithm currently implemented in this repository.

## 1) Input construction (gene-rank token sequence)

For each cell, let:
- `G` = number of genes in the raw expression vector,
- `L` = model sequence length (`seq_len`),
- `B` = batch size.

A cell has expression vector
\[
\mathbf{e} \in \mathbb{R}^{G}.
\]
Genes are sorted by descending expression, producing a permutation \(\pi\) such that
\[
e_{\pi(1)} \ge e_{\pi(2)} \ge \cdots \ge e_{\pi(G)}.
\]
Each gene symbol is mapped to a tokenizer ID, giving a token sequence
\[
\mathbf{x} = (x_1,\dots,x_G),\quad x_i \in \{0,\dots,V-1\},
\]
where \(V\) is vocabulary size. The sequence is truncated/padded to length \(L\):
\[
\tilde{\mathbf{x}} \in \{0,\dots,V-1\}^{L}.
\]
A training batch is
\[
\mathbf{X} \in \mathbb{N}^{B \times L}.
\]

## 2) Embedding and backbone

Let model width be \(d\) (`d_model`) and number of mixer layers be \(N\) (`mamba_layer`).

Embedding table:
\[
\mathbf{E} \in \mathbb{R}^{V \times d}.
\]
Token embedding output:
\[
\mathbf{H}^{(0)} = \mathbf{E}[\mathbf{X}] \in \mathbb{R}^{B \times L \times d}.
\]

For each layer \(\ell=1,\dots,N\), the mixer computes forward and reverse streams:
\[
\mathbf{F}^{(\ell)} = f_\ell(\mathbf{H}^{(\ell-1)}),\quad
\mathbf{R}^{(\ell)} = f_\ell(\operatorname{Flip}(\mathbf{H}^{(\ell-1)})),
\]
and flips back the reverse stream, then aggregates by `mode`:

- **mean**:
\[
\mathbf{H}^{(\ell)} = \tfrac{1}{2}\left(\mathbf{F}^{(\ell)}+\mathbf{R}^{(\ell)}\right)
\]
- **sum**:
\[
\mathbf{H}^{(\ell)} = \mathbf{F}^{(\ell)}+\mathbf{R}^{(\ell)}
\]
- **concat**:
\[
\mathbf{H}^{(\ell)} = \mathbf{W}_a\,[\mathbf{F}^{(\ell)}\,\|\,\mathbf{R}^{(\ell)}] + \mathbf{b}_a,
\quad \mathbf{W}_a\in\mathbb{R}^{d\times 2d}
\]
- **gate**:
\[
\mathbf{Z}^{(\ell)} = \sigma\!\left(\mathbf{W}_a[\mathbf{F}^{(\ell)}\,\|\,\mathbf{R}^{(\ell)}]+\mathbf{b}_a\right),
\]
\[
\mathbf{H}^{(\ell)} = \mathbf{Z}^{(\ell)}\odot\mathbf{F}^{(\ell)} + (1-\mathbf{Z}^{(\ell)})\odot\mathbf{R}^{(\ell)}.
\]

Then final normalization and LM head:
\[
\hat{\mathbf{H}} = \operatorname{RMSNorm}(\mathbf{H}^{(N)}) \in \mathbb{R}^{B\times L\times d},
\]
\[
\mathbf{Z}_{\text{lm}} = \hat{\mathbf{H}}\mathbf{W}_{\text{lm}}^\top + \mathbf{b}_{\text{lm}}
\in \mathbb{R}^{B\times L\times V},
\]
with \(\mathbf{W}_{\text{lm}}\in\mathbb{R}^{V\times d}\).

## 3) Autoregressive language-model loss

Using one-token shift:
\[
\mathcal{L}_{\text{LM}}=
\operatorname{CE}\left(
\mathbf{Z}_{\text{lm}}[:,0:L-1,:],
\mathbf{X}[:,1:L]
\right).
\]
Equivalent scalar form:
\[
\mathcal{L}_{\text{LM}} = -\frac{1}{B(L-1)}\sum_{b=1}^{B}\sum_{t=1}^{L-1}
\log p(x_{b,t+1}\mid x_{b,1:t}).
\]

## 4) Co-expression contrastive term (InfoNCE-like)

Let hidden states be flattened:
\[
\mathbf{h}_n \in \mathbb{R}^d,\quad n=1,\dots,BL,
\]
then L2-normalized: \(\bar{\mathbf{h}}_n = \mathbf{h}_n / \lVert\mathbf{h}_n\rVert_2\).

Remove PAD/UNK tokens, yielding valid token IDs \(g_n\) and embeddings \(\bar{\mathbf{h}}_n\).
A sparse co-expression graph provides pairs \((i,j)\) with label
\[
y_{ij}\in\{0,1\}.
\]
For each selected pair:
\[
s_{ij}=\bar{\mathbf{h}}_i^\top\bar{\mathbf{h}}_j \in [-1,1],
\quad
\ell_{ij}=s_{ij}/\tau
\]
with temperature \(\tau=0.1\).

If \(\mathcal{P}=\{(i,j): y_{ij}=1\}\) and \(\mathcal{A}\)=all selected pairs, implemented loss is:
\[
\mathcal{L}_{\text{NCE}} = -\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
\left(
\ell_{ij} - \log\sum_{(u,v)\in\mathcal{A}}e^{\ell_{uv}}
\right).
\]

## 5) Final objective

With weighting \(\gamma=0.1\):
\[
\mathcal{L} = \mathcal{L}_{\text{LM}} + \gamma\,\mathcal{L}_{\text{NCE}}.
\]

## 6) Downstream classification heads

For a cell-level classifier, hidden states \(\hat{\mathbf{H}}\in\mathbb{R}^{B\times L\times d}\) are pooled:
\[
\mathbf{c}_b = \frac{1}{L}\sum_{t=1}^{L}\hat{\mathbf{H}}_{b,t,:} \in \mathbb{R}^{d}.
\]
Then MLP classifier maps \(\mathbf{c}_b\to\mathbb{R}^{C}\), where \(C\) is class count.

---

### Implementation note
In `EncoderLayer.forward`, the code computes `output = self.mamba(X) + X` but returns `X` directly. So, strictly speaking, each layer currently behaves as identity in the present implementation.
