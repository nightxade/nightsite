# Chapter 7: Introduction to Rings
## 7.1 Basic Definitions and Examples
**Definition:**
- A *ring* $R$ is a set together with two binary operations $+,\times$, satisfying these axioms:
- $(R,+)$ is an abelian group.
- $\times$ is associative
- the *distributive laws* hold in $R$ both ways.
- The ring $R$ is *commutative* if multiplication is commutative.
- The ring $R$ is said to have an *identity* if $\exists 1\in R$ such that $1\times a=a\times1=a$, $\forall a\in R$.

**Definition:** A ring $R$ with identity $1$ with $1\neq 0$ is called a *division ring* or *skew field* if $\exists a^{-1},\forall a\in R$ with $a\neq 0$, such that $aa^{-1}=a^{-1}a=1$. A *commutative* division ring is called a *field*.
**Proposition 1:** Let $R$ be a ring and $a,b\in R$. Then
- $0a=a0=0$.
- $(-a)b=a(-b)=-(ab)$.
- $(-a)(-b)=ab$.
- If $R$ has an identity, then the identity is unique and $-a=(-1)a$.

**Definition:** Let $R$ be a ring.
- A nonzero element $a\in R$ is called a *zero divisor* if there is a nonzero element $b\in R$ such that *either* $ab=0$ or $ba=0$.
- Assume $R$ has nonzero identity $1$. An element $u\in R$ is called a *unit* if $\exists v\in R$ such that $uv=vu=1$. The set of units in $R$ is denoted $R^\times$, and is a group under multiplication. Note also that a field is a commutative $F$ with nonzero identity $1$ in which every nonzero element is a unit, i.e. $F^\times=F-\{ 0 \}$. Also, <u>a zero divisor can never be a unit</u>.

**Definition:** A commutative ring with identity $1 \neq 0$ is called an *integral domain* if it has no zero divisors ($\mathbb{Z}$-like).
**Proposition 2:** Assume $a,b,c\in R$ with element $a$ not a zero divisor. If $ab=ac$, then either $a=0$ or $b=c$. In particular, if $R$ is an integral domain, $ab=ac\implies a=0$ or $b=c$.
**Corollary 3:** Any finite integral domain is a field. A result that follows is a finite division ring is necessarily commutative, i.e. a field. This won't be proven.
**Definition:** (*Subring Criterion*) A *subring* of the ring $R$ is a subgroup of $R$ that is closed under multiplication. The subring criterion is showing that it is *nonempty* and *closed under subtraction and under multiplication*.
## 7.2 Examples: Polynomial Rings, Matrix Rings, and Groups Rings
**Polynomial Rings:** $R\in R[x]$ as the *constant polynomials*. Addition is done component-wise in $R[x]$. $R[x]$ is a *commutative ring with identity*.
**Proposition 4:** Let $R$ be an integral domain and let nonzero $p(x),q(x)\in R[x]$. Then
- $\mathrm{deg\;}p(x)q(x)=\mathrm{deg\;}p(x)+\mathrm{deg\;}q(x)$.
- the units of $R[x]$ are the units of $R$.
- $R[x]$ is an integral domain.

**Matrix Rings:** $M_{n}(R)$ is the set of all $n\times n$ matrices with entries from $R$. If $R$ is nontrivial and $n\geq2$ then $M_{n}(R)$ is *non-commutative*. An element $(a_{ij})\in M_{n}(R)$ is called a *scalar matrix* if it is diagonal, and the set of scalar matrices is a subring of $M_{n}(R)$ and is isomorphic to $R$. If $R$ has an identity $1$, then the identity $I\in M_{n}(R)$ and the subgroup of units in $M_{n}(R)$ is $GL_{n}(R)$. If a ring $S\subseteq R$ then $M_{n}(S)\subseteq M_{n}(R)$ (and is also a subring).
**Group Rings:** Fix a commutative ring $R$ with nonzero identity $1$ and let $G$ be a finite group $\{ g_{1},\dots,g_{n} \}$. The *group ring* $RG$ of $G$ with coefficients in $R$ to be the set of all formal sums $a_{1}g_{1}+\dots a_{n}g_{n},\;a_{i}\in R$. Multiplication is defined by $(ag_{i})(bg_{j})=(ab)g_{k}$, where $g_{i}g_{j}=g_{k}$. $RG$ is commutative $\iff$ $G$ is abelian. If $S\subseteq R$ then $SG\subseteq RG$. If $H\leq G$ then $RH\subseteq RG$. The set of all elements of $RG$ whose coefficients sum to zero is a subring without identity. If $\lvert G \rvert>1$, the set of elements with zero "constant term" (coefficient of identity in $G$ is $0$) is *not* a subring (not closed under multiplication).
## 7.3 Ring Homomorphisms and Quotient Rings
**Definition:** Let $R,S$ be rings.
- A *ring homomorphism* is a map $\varphi:R\to S$ with
- $\varphi(a+b)=\varphi(a)+\varphi(b)$
- $\varphi(ab)=\varphi(a)\varphi(b)$.
- $k\in\mathrm{ker\;\varphi}\implies \varphi(k)=0\in S$.
- A bijective ring homomorphism is an *isomorphism*.

**Proposition 5:** Let $R,S$ be rings and $\varphi:R\to S$ be a homomorphism.
- $\varphi(R)\leq S$.
- $\mathrm{ker\;\varphi}\leq R$. If $k\in \mathrm{ker\;\varphi}$ then $rk,kr\in \mathrm{ker\;\varphi},\forall r\in R$.

**Definition:** Let $R$ be a ring, $I\subseteq R$, and $r\in R$.
- $rI=\{ ra\mid a\in I \}$ and $Ir=\{ ar \mid a\in I \}$.
- A subset $I$ of $R$ is a *left ideal* of $R$ if $I\leq R$, $rI\subseteq I,\forall r\in R$. A *right ideal* is defined similarly.
- A subset $I$ that is a left and right ideal is called an *ideal* or *two-sided ideal*.

**Fact:** (*Ideal Criterion*) $I$ is an ideal if it is nonempty, closed under subtraction, and closed under multiplication by **all** elements of $R$.
**Proposition 6:** Let $R$ be a ring and let $I$ be an ideal of $R$. Then the additive quotient $R/I$ is a ring under the binary operations $(r+I)+(s+I)=(r+s)+I$ and $(r+I)\times(s+I)=(rs)+I$. The converse holds true too. $R/I$ is a *quotient ring*.
**Theorem 7:**
- (*First Isomorphism Theorem*) If $\varphi:R\to S$ is a homomorphism of rings, $\mathrm{ker\;\varphi}$ is an ideal of $R$, $\varphi(R)\leq S$, and $R/\mathrm{ker\;\varphi}\cong \varphi(R)$.
- If $I$ is an ideal of $R$, the map $R\to R/I$ is a surjective ring homomorphism with kernel $I$, and is called the *natural projection*. Thus, a subring is an ideal $\iff$ it is the kernel of a ring homomorphism.

**Theorem 8:** Let $R$ be a ring.
- (*Second Isomorphism Theorem*) Let $A\leq R$ and $B$ be an ideal of $R$. Then $A+B=\{ a+b\mid a\in A,b\in B \}\leq R$, $A\cap B$ is an ideal of $A$, and $(A+B)/B\cong A/(A\cap B)$.
- (*Third Isomorphism Theorem*) Let $I,J$ be ideals of $R$ with $I\subseteq J$. Then $J/I$ is an ideal of $R/I$ and $(R/I)/(J/I)\cong R/J$.
- (*Fourth/Lattice Isomorphism Theorem*) Let $I$ be an ideal of $R$. The correspondence $A\leftrightarrow A/I$ is an inclusion preserving bijection between the set of subrings $A$ of $R$ that contain $I$ and the set of subrings of $R/I$. And, $A$ is an ideal of $R$ $\iff$ $A/I$ is an ideal of $R/I$.

**Definition:** Let $I,J$ be ideals of $R$. $I+J=\{ a+b\mid a\in I,b\in J \}$, $IJ=\left\{  \sum ab\mid a\in I,b\in J  \right\}$. $I^{n}=\left\{  \sum a_{1}\dots a_{n} \mid a_{i}\in I,\leq i\leq n  \right\}$.
## 7.4 Properties of Ideals
**Definition:** Let $A\subseteq R$.
- Let $(A)$ denote the smallest ideal of $R$ containing $A$, called the *ideal generated by $A$*.
- Let $RA=\{ r_{1}a_{1}+\dots r_{n}a_{n}\mid r_{i}\in R,a_{i}\in A \}$, and $AR$ and $RAR$ be defined similarly. ($RA=0$ if $A=\emptyset$).
- An ideal generated by a single element is a *principal ideal* (like a cyclic group).
- An ideal generated by a finite set is a *finitely generated ideal*.
- If $R$ is commutative then $RA=AR=RAR=(A)$.

**Proposition 9:** Let $I$ be an ideal of $R$.
- $I=R\iff I$ contains a unit.
- Assume $R$ is commutative. $R$ is a field $\iff$ its only ideals are $0$ and $R$.

**Corollary 10:** If $R$ is a field then any nonzero ring homomorphism from $R$ into another ring is an injection.
**Definition:** An ideal $M$ in a ring $S$ is *maximal* if $M \neq S$ and the only ideals containing $M$ are $M$ and $S$.
**Proposition 11:** In a ring with identity every proper ideal is contained in a maximal ideal.
**Proposition 12:** Assume $R$ is commutative. The ideal $M$ is maximal $\iff$ $R/M$ is a field.
**Definition:** Assume $R$ is commutative. An ideal $P$ is *prime* if $P \neq R$ and whenever $ab\in P$ when $a,b\in R$, at least one of $a,b\in P$.
**Proposition 13:** Assume $R$ is commutative. Then the ideal $P$ is a prime ideal in $R$ $\iff$ $R/P$ is an integral domain.
**Corollary 14:** Assume $R$ is commutative. Every maximal ideal of $R$ is prime
## 7.5 Rings of Fractions
**Theorem 15:** Let $R$ be a commutative ring. Let $D$ be any nonempty subset of $R$ that does not contain $0$, does not contain any zero divisors, and is closed under multiplication (i.e., $ab\in D,\;\forall a,b\in D$). Then there is a commutative ring $Q$ with $1$ such that $R\leq Q$ and $\forall d\in D$, $d$ is a unit in $Q$. The ring $Q$ has the following additional properties.
- Every element of $Q$ is of the form $rd^{-1}$ for some $r\in R$ and $d\in D$. In particular, if $D=R-\{ 0 \}$, then $Q$ is a field.
- (uniqueness of $Q$) The ring $Q$ is the *smallest* ring containing $R$ in which all elements of $D$ become units: Let $S$ be a commutative ring with identity and $\varphi:R\to S$ be an injective ring homomorphism such that $\varphi(d)$ is a unit in $S$, $\forall d\in D$. Then $\exists \varPhi:Q\to S$ that is an injective homomorphism such that $\varPhi \mid_{R}=\varphi$. That is, any ring containing some $R'\cong R$ in which all elements of $D$ become units must also contain some $Q'\cong Q$.

**Definition:** Let $R,D,Q$ be as in Theorem 15.
- $Q$ is denoted the *ring of fractions* of $D$ with respect to $R$ and is denoted $D^{-1}R$.
- If $R$ is an integral domain and $D=R-\{ 0 \}$, $Q$ is the *field of fractions* or *quotient field* of $R$.

**Fact:** If $A\subseteq F$, the intersection of all subfields of $F$ containing $A$ is a subfield of $F$ and is denoted the subfield *generated* by $A$. This subfield is the smallest subfield of $F$ containing $A$, where size is defined analogously as in Theorem 15.
**Corollary 16:** Let $R$ be an integral domain and let $Q$ be the field of fractions of $R$. If a field $F$ contains a subring $R'\cong R$ then the subfield of $F$ generated by $R'$ is isomorphic to $Q$.
## 7.6 The Chinese Remainder Theorem
**Definition:** The ideals $A,B$ of the ring $R$ are *comaximal* if $A+B=R$.
**Theorem 17:** (*Chinese Remainder Theorem*) Let $A_{1},\dots,A_{k}$ be ideals in $R$. The map $R\to R/A_{1}\times\dots \times R/A_{k}$ defined by $r\mapsto(r+A_{1},\dots,r+A_{k})$ is a ring homomorphism with kernel $\bigcap_{i=1}^kA_{i}$. If $\forall i,j\in \{ 1,\dots,k \}$ with $i \neq j$ we have $A_{i}+A_{j}=R$, then the map is surjective and $\bigcap_{i=1}^kA_{i}=A_{1}\dots A_{k}=\prod_{i=1}^kA_{i}$. Thus, $R/(A_{1}\dots A_{k})=R/(A_{1}\cap\dots \cap A_{k})\cong R/A_{1}\times \dots \times R/A_{k}$.
**Corollary 18:** Let $n\in \mathbb{Z}^+$ and let $p_{1}^{a_{1}}\dots p_{k}^{a_{k}}$ be its prime factorization. Then $\mathbb{Z}/n\mathbb{Z}\cong(\mathbb{Z}/p_{1}^{a_{1}}\mathbb{Z})\times\dots \times(\mathbb{Z}/p_{k}^{a_{k}}\mathbb{Z})$ as rings. In particular, for multiplicative groups, $(\mathbb{Z}/n\mathbb{Z})^{\times}\cong(\mathbb{Z}/p_{1}^{a_{1}}\mathbb{Z})^{\times}\times\dots \times (\mathbb{Z}/p_{k}^{a_{k}}\mathbb{Z})^{\times}$.
<div style="page-break-after: always;"></div>