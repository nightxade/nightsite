# Chapter 4: Group Actions
## 4.1 Group Actions and Permutation Representations
**Definition:** The *kernel* and *stabilizer* definitions follow similarly from 2.2. An action is *faithful* if its kernel is the identity.
**Definition:** The kernel of an action is the kernel of its permutation representation. In particular, an action of $G$ on $A$ induces a faithful action of $G/\mathrm{ker\;\varphi}$ on $A$, since $g,h\in G$ induce the same permutation on $A\iff g,h$ are representatives of the same coset of $\mathrm{ker\;\varphi}$. Note that $\forall a\in A$, the kernel of the action is contained in $G_{a}$, i.e. $\mathrm{ker\;\varphi}=\bigcap_{a\in A}G_{a}$. Conversely, given a set $A\neq \emptyset$ and any homomorphism $\varphi$ of the group $G$ into $S_{A}$ we obtain an action of $G$ on $A$ such that $g\cdot a=\varphi(g)(a)$. The kernel of this action is the same as $\mathrm{ker\;\varphi}$. The permutation representation associated to this action is precisely the given homomorphism $\varphi$.
**Proposition 1:** For any group $G$ and any nonempty set $A$ there is a bijection between the actions of $G$ on $A$ and the homomorphisms of $G$ into $S_{A}$.
**Definition:** If $G$ is a group, a *permutation representation* of $G$ is any homomorphism of $G$ into the symmetric group $S_{A}$ for some nonempty set $A$. We shall say a given action of $G$ on $A$ affords or induces the associated permutation representation of $G$.
**Proposition 2:** Let $G$ be a group acting on $A\neq \emptyset$. The relation on $A$ defined by $a\sim b\iff a=g\cdot b$ for some $g\in G$ is an equivalence relation. $\forall a\in A$, the number of elements in the equivalence class containing $a$ is $\lvert G:G_{a} \rvert$.
**Definition:** Let $G$ be a group acting on $A\neq \emptyset$. The equivalence class $\{ g\cdot a\mid g\in G \}$ is the *orbit* of $G$ containing $a$. The action of $G$ on $A$ is called *transitive* if there is only one orbit for the whole set $A$, i.e., $\forall a,b\in A$, $\exists g\in G$ such that $a=g\cdot b$.
**Definition:** Subgroups of symmetric groups are called *permutation groups*.
## 4.2 Groups Acting on Themselves by Left Multiplication (*Cayley's Theorem*)
**Theorem 3:** Let $G$ be a group, $H\leq G$, and let $G$ act by left multiplication on the set $A$ of left cosets of $H$ in $G$. Let $\pi_{H}$ be the associated permutation representation afforded by this action. Then, $G$ acts transitively on $A$; $G_{1H}$ is the subgroup $H$; the action's kernel (i.e. $\mathrm{ker\;}\pi_{H}$) is $\bigcap_{x \in G}xHx ^{-1}$ and $\mathrm{ker\;}\pi_{H}$ is the largest normal subgroup of $G$ contained in $H$.
**Corollary 4:** (*Cayley's*) A group $G$ of order $n$ is isomorphic to a subgroup of $S_{n}$.
**Corollary 5:** If $G$ is a finite group of order $n$ and $p$ is the smallest prime such that $p \mid n$, then any subgroup of index $p$ is normal. (Note, however, that a group of order $n$ does not necessarily have a subgroup of index $p$).
## 4.3 Groups Acting on Themselves by Conjugation (*Class Equation*)
**Definition:** Two elements $a,b\in G$ are *conjugate* if $\exists g\in G$ such that $b=gag^{-1}$, i.e. they are in the same orbit of $G$ acting on itself by conjugation. These orbits are called **conjugacy classes**.
**Definition:** Two subsets $S,T\subseteq G$ are *conjugate* if $\exists g\in G$ such that $T=gSg^{-1}$.
**Proposition 6:** The number of conjugates of a subset $S\subseteq G$ (order of its orbit) is $\lvert G:N_{G}(S) \rvert$. The number of conjugates of an element $s \in G$ is $\lvert G:C_{G}(s) \rvert$.
**Theorem 7:** (*Class Equation*) Let $G$ be a finite group and $g_{1},g_{2},\dots,g_{r}$ be *representatives* of the distinct conjugacy classes of $G$ that are not in the center $Z(G)$. Then $\lvert G \rvert=\lvert Z(G) \rvert+\sum_{i=1}^{r}\lvert G:C_{G}(G_{i}) \rvert$.
**Theorem 8:** If $p$ is a prime and $P$ is a group of order $p^{\alpha}$ then $Z(P)\neq 1$, i.e. is nontrivial.
**Corollary 9:** If $\lvert P \rvert=p^{2}$ for some prime $p$, then $P$ is abelian, and either $P\cong Z_{p^{2}}$ or $P\cong Z_{p}\times Z_{p}$.
**Proposition 10:** Let $\sigma,\tau \in S_{n}$ and let $\sigma$ have cycle decomposition $(a_{1}\;\dots\;a_{k_{1}})(b_{1}\;\dots\;b_{k_{2}})\dots$. Then $\tau\sigma \tau ^{-1}$ has cycle decomposition $(\tau(a_{1})\;\dots\;\tau(a_{k_{1}}))(\tau(b_{1})\;\dots\;\tau(b_{k_{2}}))\dots$.
**Definition:** If $\sigma \in S_{n}$ is the product of disjoint cycles of lengths $n_{1},\dots,n_{r}$ with $n_{1}\leq \dots\leq n_{r}$, then $n_{1},\dots,n_{r}$ is the **cycle type** of $\sigma$. If $n\in \mathbb{Z}^+$. A **partition** of $n$ is any nondecreasing sequence of positive integers whose sum is $n$.
**Proposition 11:** Two elements of $S_{n}$ are conjugate in $S_{n}$ $\iff$ they have the same cycle type.
**Fact:** If $H\trianglelefteq G$, then $\forall \mathcal{K}$ conjugacy classes of $G$, either $\mathcal{K}\subseteq H$ or $\mathcal{K}\cap H=\emptyset$.
**Theorem 12:** $A_{5}$ is a simple group.
**Fact:** Conjugation is often written as a right group action with notation $a^{g}=g^{-1}ag$.
## 4.4 Automorphisms
**Definition:** Let $G$ be a group. An isomorphism from $G$ onto itself is called an **automorphism** of $G$, the group of which is denoted $\mathrm{Aut}(G)$. $\mathrm{Aut}(G)\leq S_{G}$.
**Proposition 13:** Let $H\trianglelefteq G$. Then $G$ acts by conjugation on $H$ as automorphisms of $H$, i.e. the action of $G$ on $H$ by conjugation is an automorphism $\forall g\in G$. The permutation representation afforded by this action is a homomorphism of $G$ into $\mathrm{Aut}(H)$ with kernel $C_{G}(H)$. In particular, $G/C_{G}(H)\cong F$ such that $F\leq \mathrm{Aut}(H)$.
**Fact:** Automorphisms must send subgroups to subgroups of the same order, elements of order $n$ to elements of order $n$, etc.
**Corollary 14:** If $K\leq G$ and $g\in G$, then $K\cong gKg^{-1}$. Conjugate elements and conjugate subgroups have the same order.
**Corollary 15:** $\forall H\leq G$, $N_{G}(H)/C_{G}(H)\cong F$ such that $F\leq \mathrm{Aut}(H)$. In particular, $G/Z(G)$ is isomorphic to a subgroup of $\mathrm{Aut}(G)$ (consider $H=G$).
**Definition:** Let $G$ be a group and $g\in G$. Conjugation by $g$ is called an **inner automorphism** of $G$ and the subgroup of $\mathrm{Aut}(G)$ containing all inner automorphisms is denoted by $\mathrm{Inn}(G)$.
**Definition:** Let $H\leq G$. Then $H$ is called **characteristic** in $G$, denoted $H \mathrm{\;char\;}G$, if $\forall\sigma \in \mathrm{Aut}(G)$ we have $\sigma(H)=H$.
**Fact:**
- Characteristic subgroups are normal.
- If $H$ is the unique subgroup of $G$ for a given order, then $H\mathrm{\;char\;}G$.
- If $K\mathrm{\;char\;}H$ and $H\trianglelefteq G$, then $K\trianglelefteq G$, i.e. normality is transitive in this case.

**Proposition 16:** $\mathrm{Aut}(Z_{n})\cong(\mathbb{Z}/n\mathbb{Z})^\times$, where $(\mathbb{Z}/n\mathbb{Z})^{\times}$ is an abelian group of order $\varphi(n)$.
**Fact:** Let $G$ be a group with $\lvert G \rvert=pq$, with $p,q$ prime, $p<q$, and $p\nmid q-1$. Then, $G$ is abelian.
**Proposition 17:**
- If $p$ is an odd prime and $n\in \mathbb{Z}^{+}$, $\mathrm{Aut}(Z_{p^{n}})$ is cyclic with order $\varphi(p^{n})=p^{n-1}(p-1)$.
- $\forall n\geq 3$, $\mathrm{Aut}(Z_{2^{n}})\cong Z_{2}\times Z_{2^{n-2}}$, and is not cyclic but has a cyclic subgroup of index $2$.
- Let $p$ be prime and let $V$ be an abelian group (written additively) with $pv=0,\forall v\in V$. If $\lvert V \rvert=p^{n}$, then $V$ is an $n$-dimensional vector space over the field $\mathbb{F}_{p}=\mathbb{Z}/p\mathbb{Z}$, and $\mathrm{Aut}(V)\cong GL(V)\cong GL_{n}(\mathbb{F}_{p})$.
- $\forall n\neq 6$, $\mathrm{Aut}(S_{n})=\mathrm{Inn}(S_{n})\cong S_{n}$. For $n=6$, $\lvert \mathrm{Aut}(S_{6}):\mathrm{Inn}(S_{6}) \rvert=2$.
- $\mathrm{Aut}(D_{8})\cong D_{8}$ and $\mathrm{Aut}(Q_{8})\cong S_{4}$.

**Fact:** The $V$ described in Prop. 17 is called the **elementary abelian group** of order $p^{n}$. For any prime $p$, $V_{p^{2}}=Z_{p}\times Z_{p}$, and $\mathrm{Aut}(V_{p^{2}})\cong GL_{2}(\mathbb{F}_{p})$ has order $p(p-1)^{2}(p+1)$. Corr. 9 thus implies that $\lvert P \rvert=p^{2}\implies \lvert \mathrm{Aut}(P) \rvert=p(p-1)\text{ or }p(p-1)^{2}(p+1)$, dependent on if $P$ is cyclic or elementary abelian, respectively.
## 4.5 Sylow's Theorem
**Theorem 18:** (*Sylow's*) Let $G$ be a group of order $p^{\alpha}m$, where $p$ is a prime with $p \nmid m$. Then,
- $Syl_{p}(G) \neq \emptyset$.
- If $P\in Syl_{p}(G)$ and $Q$ is any $p$-subgroup of $G$, $\exists g\in G$ such that $Q\leq gPg^{-1}$. In particular, any two Sylow $p$-subgroups of $G$ are conjugate in $G$.
- $n_{p}=\lvert Syl_{p}(G) \rvert\equiv 1 \; (\text{mod} \; p )$. Moreover, $n_{p}=\lvert G:N_{G}(P) \rvert$ for any $P\in Syl_{p}(G)$, and thus $n_{p} \mid m$.

**Lemma 19:** Let $P\in Syl_{p}(G)$. If $Q$ is any $p$-subgroup of $G$, then $Q\cap N_{G}(P)=Q\cap P$.
**Corollary 20:** Let $P$ be a Sylow $p$-subgroup of $G$. Then the following are equivalent:
- $Syl_{p}(G)=\{ P \}$, i.e. $n_{p}=1$.
- $P\trianglelefteq G$.
- $P\mathrm{\;char\;}G$.
- All subgroups generated by elements of $p$-power order are $p$-groups, i.e., if $X\subseteq G$ with $\lvert x \rvert$ being a power of $p$ for all $x \in X$, then $\langle X \rangle$ is a $p$-group.

**Facts:**
- $\lvert G \rvert=pq$ and $p \nmid q - 1$ $\implies$ abelian. If $P\in Syl_{p}(G)$ is normal in $G$, then $G$ is cyclic.
- $\lvert G \rvert=30\implies \exists H\trianglelefteq G$ with $H\cong Z_{15}$.
- $\lvert G \rvert=12\implies n_{3}=1$ or $G\cong A_{4}$.
- $\lvert G \rvert=p^{2}q\implies \exists H\trianglelefteq G$ with $\lvert H \rvert=p$ or $q$.

**Proposition 21:** If $\lvert G \rvert = 60$ and $\lvert Syl_{5}(G) \rvert > 1$, then $G$ is simple.
**Corollary 22:** $A_{5}$ is simple.
**Proposition 23:** If $G$ is a simple group of order $60$, then $G\cong A_{5}$.
(4.6) **Theorem 24:** $A_{5}$ is simple $\forall n\geq 5$.
<div style="page-break-after: always;"></div>