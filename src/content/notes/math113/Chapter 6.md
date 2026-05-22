# Chapter 6: Further Topics in Group Theory
## 6.1 $p$-groups, Nilpotent Groups, and Solvable Groups
**Definition:** A *maximal subgroup* of a group $G$ is a proper subgroup $M$ of $G$ such that $\nexists H\leq G$ with $M<H<G$.
**Theorem 1:** Let $p$ be prime and $P$ have order $p^{a},a>1$. Then
- $Z(P) \neq 1$.
- If $H$ is a nontrivial normal subgroup of $P$ then $H\cap Z(P)\neq 1$. In particular, every normal subgroup of order $p$ is contained in $Z(P)$.
- If $H\trianglelefteq P$ then $\exists F\leq H$ with $\lvert F \rvert=p^{b}$ and $F\trianglelefteq P$, $\forall p^{b}$ that divide $\lvert H \rvert$. In particular, $P$ has a normal subgroup of order $p^{b}$ for every $b\in \{ 0,\dots,a \}$.
- If $H<P$ then $H<N_{P}(H)$.
- Every maximal subgroup of $P$ is of index $p$ and is normal in $P$.

**Definition:**
- $Z_{0}(G)=1,Z_{1}(G)=Z(G),Z_{i+1}(G)/Z_{i}(G)=Z(G/Z_{i}(G))$. The chain $Z_{0}(G)\leq Z_{1}(G)\leq Z_{2}(G)\leq\dots$ is the *upper central series of $G$*.
- $G$ is *nilpotent* if $Z_{c}(G)=G$ for some $c\in \mathbb{Z}$. The smallest such $c$ is the *nilpotence class* of $G$.

**Proposition 2:** Let $p$ be a prime and $P$ be a group of order $p^{a}$. Then $P$ is nilpotent of nilpotence class at most $a-1$ for $a\geq{2}$ (and class = $a$ when $a=0,1$).
**Theorem 3:** Let $G$ be a finite group, let $p_{1},\dots,p_{s}$ be the distinct primes dividing $\lvert G \rvert$ and let $P_{i}\in Syl_{p_{i}}(G)$, $1\leq i\leq s$. Then the following are equivalent:
- $G$ is nilpotent.
- If $H<G$ then $H<N_{G}(H)$.
- $P_{i}\trianglelefteq G$ for $1\leq i\leq s$.
- $G\cong P_{1}\times\dots \times P_{s}$.

**Corollary 4:** A finite abelian group is the direct product of its Sylow subgroups.
**Proposition 5:** If $G$ is a finite group such that for all $n\in \mathbb{Z}^+$ with $n \mid \lvert G \rvert$, $G$ contains at most $n$ elements $x$ satisfying $x^{n}=1$, then $G$ is cyclic.
**Proposition 6:** (*Frattini's Argument*) Let $G$ be a finite group, $H\trianglelefteq G$, and $P\in Syl_{p}(H)$. Then $G=HN_{G}(P)$ and $\lvert G:H \rvert$ divides $\lvert N_{G}(P) \rvert$.
**Proposition 7:** A finite group is nilpotent $\iff$ every maximal subgroup is normal.
**Definition:** For any (finite or infinite) group $G$ we define $G^{0}=G,G^{1}=[G,G],G^{i+1}=[G,G^{i}]$. The chain of groups $G^{0}\geq G^{1}\geq G^{2}\geq\dots$ is called the *lower central series* of $G$.
**Theorem 8:** A group $G$ is nilpotent $\iff$ $G^{n}=1$ for some $n\geq{0}$. In fact, $G$ is nilpotent of class $c$ $\iff$ $c$ is the smallest nonnegative integer such that $G^{c}=1$. Also, then $G^{c-i}\leq Z_{i}(G)$, $froalli\in \{ 0,..,c \}$.
**Definition:** For any group $G$ we define $G^{(0)}=G,G^{(1)}=[G,G],G^{(i+1)}=[G^{(i)},G^{(i)}]$. The chain of groups $G^{(0)}\geq G^{(1)}\geq\dots$ is the *derived* or *commutator* series of $G$.
**Definition:** A group $G$ is *solvable* if there exists a chain of subgroups $1=G_{0}\trianglelefteq G_{1}\trianglelefteq\dots\trianglelefteq G_{s}=G$ such that $G_{i+1}/G_{i}$ is abelian $\forall i\in \{ 0,1,\dots,s-1 \}$.
**Theorem (3.4):** A finite group $G$ is solvable $\iff$ $\forall n$ with $n \mid \lvert G \rvert$ and $\left( n, \frac{\lvert G \rvert}{n} \right)=1$, $G$ has a subgroup of order $n$.
**Theorem 9:** A group $G$ is solvable $\iff$ $G^{(n)}=1$ for some $n\geq{0}$.
**Definition:** If $G$ is solvable, the smallest nonnegative $n$ such that $G^{(n)}=1$ is the *solvable length* of $G$.
**Proposition 10:** Let $G,K$ be groups, $H\leq G$, and $\varphi:G\to K$ be a surjective homomorphism. Then
- $H^{(i)}\leq G^{(i)},\forall i\geq 0$. In particular, if $G$ is solvable, so too is $H$, and the solvable length of $H$ is $\leq$ the solvable length of $G$.
- $\varphi(G^{(i)})=K^{(i)}$. In particular, homomorphic images and quotient groups of solvable groups are solvable, with solvable length less than or equal to that of the domain group $G$.
- If $N\trianglelefteq G$ and both $N$ and $G/N$ are solvable, then so is $G$.

**Theorem 11:** Let $G$ be a finite group.
- (*Burnside*) If $\lvert G \rvert=p^{a}q^{b}$ for some primes $p,q$, then $G$ is solvable.
- (*Philip Hall*) If $\forall p$ with $p \mid \lvert G \rvert$, we factor $\lvert G \rvert=p^{a}m$ with $(p,m)=1$, and $G$ has a subgroup of order $m$, then $G$ is solvable. (i.e. $\forall p$, $G$ has a subgroup with index equal to the order of a Sylow $p$-subgroup, called a *Sylow $p$-complement*, then $G$ is solvable).
- (*Feit-Thompson*) If $\lvert G \rvert$ is odd then $G$ is solvable.
- (*Thompson*) If $\forall$ pairs of elements $x,y\in G$, $\langle x,y \rangle$ is solvable, then $G$ is solvable.

## 6.2 Applications in Groups of Medium Order
**Technique 1:** (*Counting Elements*) Consider Sylow $p$-subgroups where $p \mid \lvert G \rvert$ and $p^{2} \nmid \lvert G \rvert$. Non-normality implies $n_{p}>1,\forall p$. Calculate # of elements of prime order from this, show that it is $>G\implies$ not simple. Or, show that $n_{p}=1$ for some $p$.
**Technique 2:** (*Exploiting subgroups of small index*) This is derived from the fact that if $H\leq G$ and $\lvert G:H \rvert=k$, $\exists \varphi:G\to S_{k}$ with $\mathrm{ker\;\varphi}\leq H$. If $G$ is simple, $k=1\implies G\cong F\leq S_{k}\implies \lvert G \rvert \mid k!$ (that is, $\varphi$ must be an injective homomorphism since $\mathrm{ker\;}\varphi\trianglelefteq G$ and we assume $G$ is simple). $\lvert G \rvert={p_{1}}^{\alpha_{1}}\dots{p_{s}}^{\alpha_{s}}$ with $p_{1}<\dots<p_{s}$. Minimal possible index of a proper subgroup is (usually) dependent on $\alpha_{s}$. $\alpha_{s}=1\implies \geq p_{s}$, $\alpha_{s}=2\implies\geq 2p_{s}$.
**Technique 3:** (*Permutation Representations*) Consider a proper subgroup of index $k$ in $G$ with $\lvert G \rvert=n$. Then, $G\cong F\leq S_{k}$. We then calculate within $S_{k}$ that $S_{k}$ contains no simple subgroup of order $n$. Two helpful restrictions are
- if $G$ contains an element/subgroup of a particular order, so too must $S_{k}$
- if $P\in Syl_{p}(G)$ and $P\in Syl_{p}(S_{k})$, then $\lvert N_{G}(P) \rvert \mid \lvert N_{S_{k}}(P) \rvert$. Note that *(2)* occurs frequently when $k=p$ or $p+1$ and $G$ has a subgroup of index $k$, as it implies $\lvert N_{S_{k}}(P) \rvert=p(p-1)\implies \lvert N_{G}(P) \rvert \mid p(p-1)$. It may also help to work in $A_{k}$ instead of $S_{k}$ (see Prop 12).

**Technique 4:** (*Playing $p$-subgroups off against each other for distinct primes $p$*)
- Suppose $p,q$ are distinct primes such that every group of order $pq$ is cyclic $\implies p \nmid q-1$ with $p<q$. If $Q\in Syl_{q}(G)$ with $\lvert Q \rvert=q$ and $p\mid \lvert N_{G}(Q) \rvert$, $\exists P\leq N_{G}(Q)$ with $\lvert P \rvert=p$. Thus, $PQ$ is a group, and if $PQ$ is abelian, $PQ\leq N_{G}(P)$ and thus $q \mid \lvert N_{G}(P) \rvert$. This may force $N_{G}(P)=G\implies P\trianglelefteq G$ or at least force $N_{G}(P)$ to have index less than the minimal index $\implies$ contradiction.
- We can refine this method. Let $Q\in Syl_{q}(G)$ and $p \mid \lvert N_{G}(Q) \rvert$. Let $P\in Syl_{p}(N_{G}(Q))$. Apply Sylow's Theorem, and if $P\trianglelefteq N_{G}(Q)$, we force $N_{G}(P)$ to have small index. If $P\in Syl_{p}(G)$, we can put further restrictions on $\lvert N_{G}(P) \rvert$. If not, then Sylow's Theorem implies $P< P^{*}\in Syl_{p}(G)\implies P<N_{P^{*}}(P)\leq N_{G}(P)$.

**Technique 5:** (*Studying normalizers of intersections of Sylow $p$-subgroups*) Technique 1 fails to generalize because if distinct $P,R\in Syl_{p}(G)$ and $\lvert P \rvert=p^{\alpha},\alpha\geq 2$, it is not necessary for $P\cap R=1$. Suppose $\exists P,R$ with the above true. Let $P_{0}=P\cap R$. Then, $P_{0}<P$ and $P_{0}<R$. By Theorem 1, $P_{0}<N_{P}(P_{0})$ and $P_{0}<N_{R}(P_{0})$. We aim to use this to prove that the normalizer in $G$ has smaller index than the minimal index. This works well when $\lvert P_{0} \rvert=p^{\alpha-1}$, as then $N_{G}(P_{0})$ has two distinct Sylow $p$-subgroups $P,R$ since $P_{0}$ is a maximal subgroup of $P,R$. In particular, $\lvert N_{G}(P_{0}) \rvert=p^{\alpha }k$ where $k\geq p+1$.
**Proposition 12:**
- If $G$ has no subgroup of index $2$ and $G\leq S_{k}$, then $G\leq A_{k}$.
- If $P\in Syl_{p}(S_{k})$ for some odd prime $p$, then $P\in Syl_{p}(A_{k})$ and $\lvert N_{A_{k}}(P) \rvert=\frac{1}{2}\lvert N_{S_{k}}(P) \rvert$.

**Lemma 13:** In a finite group $G$, if $n_{p}\not\equiv1\; (\text{mod} \; p^{2} )$, then $\exists P,R\in Syl_{p}(G)$ with $P\neq R$ and $P\cap R$ is of index $p$ in both $P,R$ (and thus normal in each).
**Proposition 14:** If $G$ is a simple group of order 168, then
- $n_{2}=21,n_{3}=7,n_{7}=8$.
- Sylow $2$-subgroups of $G$ are dihedral, Sylow $3$- and $7$-subgroups are cyclic.
- $G$ is isomorphic to a subgroup of $A_{7}$ and $G$ has no subgroup of index $\leq 6$.
- The conjugacy classes of $G$ are the identity, two classes with order 24 and elements of order 7 (represented by an order 7 element and its inverse), one class with order 56 and elements of order 3, one class of order 42 and elements of order 4, and one class of order 21 and elements of order 2 (in particular, every element of $G$ has order a power of a prime).
- If $T\in Syl_{2}(G)$ and $U,W$ are the two Klein 4-groups in $T$, then $U,W$ are not conjugate in $G$ and $N_{G}(U)\cong N_{G}(W)\cong S_{4}$.
- $G$ has precisely three conjugacy classes of maximal subgroups, two $\cong S_{4}$ and one $\cong$ non-abelian group of order 21.

**Theorem 15:** Up to isomorphism there is a unique simple group of order 168, $GL_{3}(\mathbb{F}_{2})\cong$ automorphism group of the projective plane $\mathcal{F}$.
## 6.3 A Word on Free Groups
**Theorem 16:** $F(S)$ is a group under the binary operation defined as follows:
$$
(r_{1}^{\delta_{1}}r_{2}^{\delta_{2}}\dots r_{m}^{\delta_{m}})(s_{1}^{\delta_{1}}s_{2}^{\delta_{2}}\dots s_{m}^{\delta_{m}})=\left\{ \begin{matrix*}
r_{1}^{\delta_{1}}\dots r_{m-k+1}^{\delta_{m-k+1}}s_{k}^{\epsilon_{k}}\dots s_{n}^{\epsilon_{n}}, & k\leq m \\
s_{m+1}^{\epsilon_{m+1}}\dots s_{n}^{\epsilon_{n}},& k=m+1\leq n, \\
1,& k=m+1,m=n
\end{matrix*} \right.
$$
Where $k$ is the smallest integer in the range $1\leq k\leq m+1$ such that $s_{k}^{\epsilon_{k}}\neq r_{m-k+1}^{-\delta_{m-k+1}}$. (Where the product of $r$ and $s$ in the center does not cancel).
**Theorem 17:** Let $G$ be a group, $S$ a set, and $\varphi:S\to G$ a set map. Then there is a unique group homomorphism $\Phi:F(S)\to G$ such that the following diagram commutes:
```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[every label/.append style={font=\normalsize}]
S \arrow[rr, "\mathrm{inclusion}" above, midway] \arrow[rrd, "\varphi" below left=2pt, midway] &&  F(S) \arrow[d, "\Phi" right=1pt, midway] \\
&& G
\end{tikzcd}
\end{document}
```

**Corollary 18:** $F(S)$ is unique up to a unique isomorphism which is the identity map on the set $S$.
**Definition:** The group $F(S)$ is called the *free group* on the set $S$. A group $F$ is a *free group* if there is some $S$ such that $F=F(S)$ — in this case we call $S$ a set of *free generators* (or a *free basis*) of $F$. The cardinality of $S$ is the *rank* of the free group.
**Theorem 19:** (*Schreier*) Subgroups of a free group are free.
**Definition:** Let $S\subseteq G$ such that $G=\langle S \rangle$.
- A *presentation* for $G$ is a pair $\langle S,R \rangle$, where $R$ is a set of words in $F(S)$ such that the normal closure of $\langle R \rangle$ in $F(S)$ (the smallest normal subgroup containing $\langle R \rangle$) equals the $\mathrm{ker\ }\pi$, where $\pi:F(S)\to G$ is a homomorphism that extends the identity mapping from $S$ to $S$. The elements of $S$ are called *generators* and those of $R$ are called *relations*.
- We say $G$ is *finitely generated* if there is a presentation $(S,R)$ such that $S$ is a finite sets and we say $G$ is *finitely presented* if there is a presentation $(S,R)$ with both $S$ and $R$ finite sets.

<div style="page-break-after: always;"></div>