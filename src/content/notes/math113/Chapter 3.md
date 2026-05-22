# Chapter 3: Quotient Groups and Homomorphisms
## 3.1 Quotient Groups
**Proposition 1:** Let $G,H$ be groups and let $\varphi:G\to H$ be a homomorphism. Then,
$$
\varphi(1_{G})=1_{H} \qquad
\varphi(g^{-1})=\varphi(g)^{-1} \qquad
\varphi(g^{n})=\varphi(g)^{n} \qquad
\mathrm{ker\;}\varphi\leq G \qquad
\mathrm{im\;}(\varphi)\leq H
$$
**Theorem 3:** Let $G,H$ be groups and $K=\mathrm{ker\;}\varphi$ with $\varphi:G\to H$. Then the set of left cosets of $K$ in $G$ is $G/K$ and the operation $uK\circ vK=(uv)K$ is well defined if the converse of injectivity is true, i.e. $u=v\implies \varphi(u)=\varphi(v)$.
**Proposition 4:** Let $G$ be a group and $N\leq G$. Then the set of left cosets of $N$ in $G$ partition $G$, and $\forall u,v\in G$, $uN=vN\iff v^{-1}u\in N$.
**Proposition 5:** Let $G$ be a group and $N\leq G$. The operation $uN\cdot vN=(uv)N$ is well defined $\iff gng^{-1}\in N,\forall g\in G$ and $\forall n\in N$. If the operation if well defined, the set of left cosets of $N$ in $G$ is a group.
**Definition:** $gng^{-1}$ is the conjugate of $n\in N$ by $g\in G$. The set $gNg^{-1}=\{ gng^{-1} \mid n\in N \}$ is called the *conjugate* of $N$ by $g$. The element $g$ *normalizes* $N$ if $gNg^{-1}=N$. A subgroup $N$ of $G$ is normal ($N\trianglelefteq G$) if $gNg^{-1}=N,\forall g\in G$. All subgroups of an *abelian* group are normal.
**Theorem 6:** Let $N\leq G$. The following statements are equivalent:
$$
N\trianglelefteq G \qquad
N_{G}(N)=G \qquad
gN=Ng, \forall g\in G \qquad
\text{The set of left cosets of }N\text{ in }G\text{ are a group} \qquad
gNg^{-1}\subseteq N,\forall g\in G
$$
**Proposition 7:** $N\trianglelefteq G\iff \exists \varphi$ such that $\varphi$ is a homomorphism and $\mathrm{ker\;\varphi}=N$.
**Definition:** Let $N\trianglelefteq G$. The homomorphism $\pi:G\to G/N$ defined by $\pi(g)=gN$ is a natural projection/homomorphism. If $\overline{H}\leq G/N$, the complete preimage of $\overline{H}$ in $G$ is the preimage of $\overline{H}$ under the natural projection
**Fact:** The quotient groups of of a cyclic group are cyclic with order $\frac{\lvert G \rvert}{\lvert N \rvert}$.
## 3.2 Lagrange's Theorem
**Theorem 8:** (Lagrange) If $G$ is a finite group and $H\leq G$, then $\lvert  H \rvert \Big| \lvert G \rvert$ and $\lvert G:H \rvert=\frac{\lvert G \rvert}{\lvert H \rvert}$. This also implies $\frac{\lvert G:H \rvert}{\lvert H:K \rvert}=\lvert G:K \rvert$.
**Corollary 9:** If $G$ is a finite group and $x \in G$, $\lvert  x \rvert = \lvert \langle x \rangle \rvert \Big| \lvert G \rvert$ and $x^{\lvert G \rvert} =1$.
**Corollary 10:** If $G$ is a group of prime order $p$, then $G$ is cyclic $\implies G\cong Z_{p}$.
**Theorem 11:** (*Cauchy*) If $G$ is a finite group and $p \Big| \lvert G \rvert$, then $\exists x \in G$ such that $\lvert x \rvert=p$.
**Theorem 12:** (*Sylow*) If $G$ is a finite group of order $p^{\alpha}m$, where $p$ is a prime and $p \nmid m$, then $\exists H\leq G$ such that $\lvert H \rvert=p^{\alpha}$.
**Definition:** Let $H,K\leq G$. Then $HK=\{ hk \mid h\in H, k\in K \}$.
**Proposition 13:** If $H,K\leq G$ and are finite, then $\lvert HK \rvert=\frac{\lvert H \rvert\lvert K \rvert}{\lvert H\cap K \rvert}$.
**Proposition 14:** If $H,K\leq G$, $HK\leq G\iff HK=KH$.
**Corollary 15:** If $H,K\leq G$ and $H\leq N_{G}(K)$, then $HK\leq G$. In particular, if $K\trianglelefteq G$ then $HK\leq G$ for any $H\leq G$.
## 3.3 Isomorphism Theorems
**Theorem 16:** (1) If $\varphi:G\to H$ is a homomorphism of groups, then $\mathrm{ker\;}\varphi \trianglelefteq G$ and $G/\mathrm{ker\;}\varphi \cong \varphi(G)$.
**Corollary 17:** Let $\varphi:G\to H$ be a homomorphism of groups. $\varphi$ is injective $\iff \mathrm{ker\;\varphi}=1$, and $\lvert G:\mathrm{ker\;\varphi} \rvert=\lvert \varphi(G) \rvert$.
**Theorem 18:** (2) Let $G$ be a group, let $A,B\leq G$ and assume $A\leq N_{G}(B)$. Then $AB$ is a subgroup of $G$, $B\trianglelefteq AB$, $A\cap B\trianglelefteq A$ and $AB/B \cong A/A\cap B$.
**Theorem 19:** (3) Let $G$ be a group and let $H,K\trianglelefteq G$ with $H\leq K$. Then $K/H\trianglelefteq G/H$ and $(G/H)/(K/H)\cong G/K$.
**Theorem 20:** (4) Let $G$ be a group and let $N\trianglelefteq G$. Then there is a bijection from the set of subgroups $A$ of $G$ which contain $N$ onto the set of subgroups $\overline{A}=A/N$ of $G/N$. In particular, every subgroup of $\overline{G}$ is of the form $A/N$ for some subgroup $A$ of $G$ containing $N$ (namely, its preimage in $G$ under the natural projection homomorphism from $G$ to $G/N$). $\forall A,B\leq G$ with $N\leq A$ and $N\leq B$, this bijection has the following properties:
$$
A\leq B\iff \overline{A}\leq \overline{B} \qquad
A\leq B\implies \lvert B:A \rvert=\lvert \overline{B}:\overline{A} \rvert \qquad
\overline{\langle A,B \rangle}=\langle \overline{A},\overline{B} \rangle \qquad
\overline{A\cap B}=\overline{A}\cap \overline{B} \qquad
A\trianglelefteq G\iff \overline{A}\trianglelefteq \overline{G}
$$
## 3.4 Composition Series and the Hölder Program
**Proposition 21:** If $G$ is a finite abelian group and $p$ is a prime dividing $\lvert G \rvert$, then $\exists x \in G$ with $\lvert x \rvert=p$.
**Definition:** A (finite or infinite) group $G$ is *simple* if $\lvert G \rvert > 1$ and the only normal subgroups of $G$ are $1$ and $G$.
**Definition:** In a group $G$ a sequence of subgroups $1=N_{0}\leq N_{1}\leq\dots\leq N_{k-1}\leq N_{k}=G$ is called a *composition series* if $N_{i}\trianglelefteq N_{i+1}$ and $N_{i+1}/N_{i}$ is a simple group for $0\leq i\leq k-1$. If the above sequence is a composition series, the quotient groups $N_{i+1}/N_{i}$ are called *composition factors* of $G$.
**Theorem 22:** (Jordan-Hölder) Let $G$ be a finite group with $G \neq 1$. Then
- $G$ has a composition series
- The composition factors in a composition series are unique, namely, if $1=N_{0}\leq N_{1}\leq\dots\leq N_{r}=G$ and $1=M_{0}\leq M_{1}\leq\dots\leq M_{s}=G$ are two composition series for $G$, then $r=s$ and there is some permutation $\pi$ of $\{ 1,2,\dots,r \}$ such that $M_{\pi(i)}/M_{\pi(i)-1}\cong N_{i}/N_{i-1}$, for $1\leq i\leq r$.

**Theorem:** There is a list consisting of 18 (infinite) families of simple groups and 26 simple groups not belonging to these families (the *sporadic* simple groups) such that every finite simple group is isomorphic to one of the groups in this list.
**Theorem:** (*Feit-Thomopson*) If $G$ is a simple group of odd order, then $G\cong Z_{p}$ for some prime $p$.
**Definition:** A group $G$ is *solvable* if there exists a chain of subgroups $1=G_{0}\trianglelefteq G_{1}\trianglelefteq \dots\trianglelefteq G_{s}=G$ such that $G_{i+1}/G_{i}$ is abelian for $i=0,1,\dots,s-1$.
**Theorem:** The finite group $G$ is solvable $\iff$ for every divisor $n$ of $\lvert G \rvert$ such that $\left( n, \frac{\lvert G \rvert}{n} \right)=1$, $G$ has a subgroup of order $n$.
## 3.5 Transpositions and the Alternating Group
**Definition:** A $2$-cycle is called a *transposition*.
**Definition:**
- $\epsilon(\sigma)$ is called the *sign of $\sigma$*.
- $\sigma$ is called an *even permutation* if $\epsilon(\sigma)=1$ and an *odd permutation* if $\epsilon(\sigma)=-1$.

**Proposition 23:** The map $\epsilon:S_{n}\to \{ \pm1 \}$ is a homomorphism (where $\{ \pm1 \}\cong Z_{2}$)
**Proposition 24:** Transpositions are all odd permutations and $\epsilon$ is surjective.
**Definition:** The *alternating group of degree $n$*, denoted by $A_{n}$, is the kernel of the homomorphism $\epsilon$ (i.e. the set of even permutations).
**Proposition 25:** The permutation $\sigma$ is odd $\iff$ the number of cycles of even length in its cycle decomposition is odd.
<div style="page-break-after: always;"></div>