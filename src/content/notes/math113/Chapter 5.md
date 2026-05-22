# Chapter 5: Direct and Semidirect Products and Abelian Groups
## 5.1 Direct Products
**Proposition 1:** If $G_{1},\dots,G_{n}$ are groups, $\lvert G_{1}\times\dots \times G_{n} \rvert=\lvert G_{1} \rvert\dots \lvert G_{n} \rvert$. Note that if $\lvert G_{i} \rvert=\infty$, for any $i$, the direct product's order is similarly infinite.
**Proposition 2:** Let $G=G_{1}\times\dots \times G_{n}$.
- For each fixed $i$, the set of elements which have the identity of $G_{j}$ in the $j$th position for all $j \neq i$ is a subgroup of $G$ isomorphic to $G_{i}\cong \{ (1,\dots,1,g_{i},1,\dots,1) \mid g_{i}\in G_{i} \}$.
- For each fixed $i$, define $\pi_{i}:G\to G_{i}$ by $\pi_{i}((g_{1},\dots,g_{n}))=g_{i}$. Then $\pi_{i}$ is a surjective homomorphism with $\mathrm{ker\;\pi_{i}}\cong G_{1}\times\dots \times G_{i-1}\times G_{i+1}\times\dots \times G_{n}$.

**Definition:** The *elementary abelian group* is $E_{p^{n}}=\underbrace{ Z_{p}\times\dots \times Z_{p} }_{ n }$, with the property that $x^{p}=1,\forall x \in E_{p^{n}}$.
## 5.2 The Fundamental Theorem of Finitely Generated Abelian Groups
**Definition:**
- A group $G$ is *finitely generated* if $\exists A\subseteq G$ with $\lvert A \rvert$ finite such that $G=\langle A \rangle$. (e.g. $\mathbb{Z}$ is finitely generated because $\mathbb{Z}=\langle 1 \rangle$).
- For each $r\in \mathbb{Z}$ with $r\geq 0$, let $\mathbb{Z}^{r}=\underbrace{ \mathbb{Z}\times\dots \times \mathbb{Z} }_{ r }$, where $\mathbb{Z}^{0}=\mathbf{1}$. The group $\mathbb{Z}^{r}$ is called the *free abelian group of rank $r$*.

**Theorem 3:** (*Chapter Title*) Let $G$ be a finitely generated abelian group.
- Then $G\cong \mathbb{Z}^{r}\times Z_{n_{1}}\times\dots \times Z_{n_{s}}$, provided that $r\geq 0$, $n_{j}\geq 2,\forall j$, and $n_{i+1} \mid n_{i}, \forall_{1}\leq i\leq s -1$.
- $G$ only has one possible (unique) representation of this form.
- $r$ is called the *free rank* or *Betti number* of $G$, the integers $n_{1},\dots,n_{s}$ are the *invariant factors* of $G$, and the representation if called the *invariant factor decomposition* of $G$.

**Fact:**
- Two finitely generated abelian groups are isomorphic $\iff$ same free rank and same list of invariant factors.
- A finitely generated abelian group is finite only if $r=0$.
- Every prime divisor of $n$ must divide the first invariant factor $n_{1}$.
- The type of $G$ is $(n_{1},\dots,n_{s})$.

**Corollary 4:** If $n$ is the product of distinct primes, then up to isomorphism  the only abelian group of order $n$ is $Z_{n}$.
**Fact:** Determining all possible isomorphism classes of abelian groups of order $n$ is solved by factoring $n$ and partitioning into invariant factors.
**Theorem 5:** Let $G$ be an abelian group of order $n>1$ and let the unique factorization of $n$ into distinct prime powers be $n=p_{1}^{\alpha_{1}}\dots p_{k}^{\alpha_{k}}$.
- Then $G\cong A_{1}\times\dots \times A_{k}$, where $\lvert A_{i} \rvert=p_{i}^{\alpha _{i}}$ (these are **not** alternating groups).
- For each $A\in \{ A_{1},..,A_{k} \}$ with $\lvert A \rvert=p^{\alpha}$, $A\cong Z_{p^{\beta_{1}}}\times\dots \times Z_{p^{\beta_{t}}}$ with $\beta_{1}\geq\dots\geq\beta_{t}\geq 1$ and $\sum_{i=1}^t \beta_{i}=\alpha$.
- $G$ has only one possible (unique) decomposition of the form in *(1)*, and each $A$ similarly has only a unique decomposition of the form in *(2)*. Note also that $Z_{p^{\beta_{1}}}\times\dots \times Z_{p^{\beta_{t}}}$ are the invariant factors of $A_{i}$.
- The integers $p^{\beta_{i}}$ are called the *elementary divisors of $G$*. The decomposition in *(1),(2)* is called the *elementary divisor decomposition* of $G$.

**Fact:** Determining all the possible invariant factor decompositions of finitely generated abelian groups of order $n$ is equivalent to determining the number of ways to partition each $\alpha$ (for each $p^{\alpha}$ in the factorization of $n$), then independently combining them.
**Proposition 6:** Let $m,n\in \mathbb{Z}^+$.
- $Z_{m}\times Z_{n}\cong Z_{mn}\iff(m,n)=1$.
- $n=p_{1}^{\alpha_{1}}\dots p_{k}^{\alpha_{k}}\implies Z_{n}\cong Z_{p_{1}^{\alpha_{1}}}\times\dots \times Z_{p_{k}^{\alpha_{k}}}$.

**Example: (Invariant Factors/Cyclic Decomposition $\to$ Elementary Divisors):** $\lvert G \rvert=1800$, $G\cong Z_{30}\times Z_{30}\times Z_{2}$ $\longrightarrow$ $E=\{ 2,2,2,3,3,5,5 \}$.
**Example: (Elementary Divisors $\to$ Invariant Factors):** $\lvert G \rvert=1800$, $E=\{ 2,2,2,3,3,25 \}$ $\longrightarrow$ $G\cong Z_{2*3*25}\times Z_{2*3*1}\times Z_{2*1*1}\cong Z_{150}\times Z_{6}\times Z_{2}$.
**Example: (Elementary Divisors):**
- $Z_{6}\times Z_{15}$ has $E=\{ 2,3,3,5 \}$ $\implies \cong Z_{2}\times Z_{3}\times Z_{3}\times Z_{5}$.
- $Z_{10}\times Z_{9}$ has $E=\{ 2,5,9 \}$ $\implies \cong Z_{2}\times Z_{5}\times Z_{9}$.

**Definition:**
- If $G$ is a finite abelian group of type $(n_{1},\dots,n_{t})$, then $t$ is called the *rank* of $G$.
- The *exponent* of $G$ is the smallest $n\in \mathbb{Z}^+$ such that $x^{n}=1,\forall x \in G$. If $\nexists n$, then the exponent is $\infty$.

## 5.4 Recognizing Direct Products
**Definition:** Let $G$ be a group, $x,y\in G$, and $A,B\subseteq G$ with $A,B\neq \emptyset$.
- $[x,y]=x^{-1}y^{-1}xy$ is called the *commutator* of $x$ and $y$.
- Define $[A,B]=\langle [a,b] \mid a\in A,b\in B \rangle$.
- Define $G'=\langle [x,y] \mid x,y\in G \rangle$ to be the *commutator subgroup* of $G$.

**Proposition 7:** Let $G$ be a group, $x,y\in G$, and $H\leq G$.
- $xy=yx[x,y]$. Importantly, $xy=yx \iff[x,y]=1$.
- $H\trianglelefteq G\iff[H,G]<H$.
- $\sigma[x,y]=[\sigma(x),\sigma(y)]$ for any $\sigma \in \mathrm{Aut}(G)$. Also, $G'\mathrm{\;char\;}G$ and $G/G'$ is abelian.
- $G/G'$ is the largest abelian quotient of $G$, i.e. $H\trianglelefteq G$ and $G/H$ is abelian $\iff G'\leq H$.
- If $\varphi:G\to A$ is any homomorphism of $G$ into an abelian group $A$, then $\varphi$ factors through $G'$, i.e. $G'\leq \mathrm{ker\;\varphi}$.

**Proposition 8:** Let $H,K\leq G$. The number of distinct ways of writing each element of the set $HK$ in the form $hk$ is $\lvert H\cap K \rvert$. If $H\cap K=1$, each element of $HK$ can be written uniquely as $hk$.
**Theorem 9:** Let $G$ be a group with $H,K\leq G$ satisfying
- $H,K\trianglelefteq G$.
- $H\cap K=1$. Then, $HK\cong H\times K$.

**Definition:** If $G$ is a group satisfying Theorem 9, $HK$ is called the *internal direct product* and $H\times K$ is called the *external direct product.*
## 5.5 Semidirect Products
**Theorem 10:** Let $H,K$ be groups and let $\varphi:K\to \mathrm{Aut}(H)$ be a homomorphism. Let $\cdot$ denote the left action of $K$ on $H$ determined by $\varphi$. Let $G$ be the set of $(h,k)$, then define the following multiplication on $G$: $(h_{1},k_{1})(h_{2},k_{2})=(h_{1}\;k_{1}\cdot h_{2},k_{1}k_{2})$. Then,
- $G$ is a group with order $\lvert H \rvert \lvert K \rvert$.
- $H\cong\{(h,1) \mid h\in H\}$ and $K\cong \{ (1,k)\mid k\in K \}$, and the isomorphic copies of $H,K$ in $G$ are subgroups of $G$.

Now, identifying $H$,$K$ with their isomorphic copies in $G$.
- $H\trianglelefteq G$.
- $H\cap K=1$.
- $\forall h\in H,\forall k\in K$, $khk^{-1}=k\cdot h=\varphi(k)(h)$.

**Proposition 11:** Let $H,K$ be groups and let $\varphi:K\to \mathrm{Aut}(H)$ be a homomorphism. Then the following are equivalent:
- $H\rtimes K\cong H\times K$
- $\varphi$ is the trivial homomorphism
- $K\trianglelefteq H\rtimes K$.

**Theorem 12:** Suppose $G$ is a group with $H,K\leq G$ such that
- $H\trianglelefteq G$
- $H\cap K=1$. Let $\varphi:K\to \mathrm{Aut}(H)$ be the homomorphism that maps $k$ to an inner automorphism. Then $HK\cong H\rtimes K$.

**Definition:** Let $H\leq G$. The *complement* for $H$ in $G$ is some $K\leq G$ with $G=HK$ and $H\cap K=1$.
<div style="page-break-after: always;"></div>