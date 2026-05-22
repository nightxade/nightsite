# Chapter 1: Introduction to Groups
## 1.1 Basic Axioms and Examples
**Definition:**
- A *binary operation* $*$ on a set $G$ is a function $*:G\times G\to G$. For any $a,b\in G$ we shall write $a*b$ for $*(a,b)$.
- A binary operation $*$ on a set $G$ is *associative* if for all $a,b,c\in G$ we have $a*(b*c)=(a*b)*c$.
- If $*$ is a binary operation on a set $G$ we say elements $a$ and $b$ of $G$ *commute* if $a*b=b*a$. We say $*$ (or $G$) is *commutative* if for all $a,b\in G$, $a*b=b*a$.

**Definition:**
- A *group* is an ordered pair $(G,*)$ where $G$ is a set and $*$ is a binary operation on $G$ satisfying the following axioms:
- $*$ is associative
- $\exists e\in G$, denoted the *identity* of $G$, such that $\forall a\in G$, $a*e=e*a=a$
- $\forall a\in G$, $\exists a^{-1}\in G$ called the *inverse* of $a$ such that $a*a^{-1}=a^{-1}*a=e$.

**Proposition 1:** If $G$ is a group under the operation $*$, then
- the identity of $G$ is unique
- $\forall a\in G$, $a^{-1}$ is uniquely determined
- $(a^{-1})^{-1}=a,\forall a\in G$
- $(a*b)^{-1}=(b^{-1})*(a^{-1})$
- for any $a_{1},\dots,a_{n}\in G$, the value of $a_{1}*a_{2}*\dots*a_{n}$ is independent of how the expression is bracketed (this is called the *generalized associative law*).

**Proposition 2:** Let $G$ be a group and let $a,b\in G$. The equations $ax=b$ and $ya=b$ have unique solutions for $x,y\in G$. In particular, the left and right cancellation laws hold in $G$, i.e.
- if $au=av$, then $u=v$
- if $ub=vb$, then $u=v$.

**Definition:** For $G$ a group and $x \in G$, define the *order* of $x$ to be the smallest positive integer $n$ such that $x^{n}=1$, and denote this integer by $\lvert x \rvert$. In this case $x$ is said to be of order $n$. If no positive power of $x$ is the identity, the order of $x$ is defined to be infinity and $x$ is said to be of infinite order.
**Definition:** Let $G=\{ g_{1},g_{2},\dots,g_{n} \}$ be a finite group with $g_{1}=1$. The *multiplication table* or *group table* is the $n\times n$ matrix whose $i,j$ entry is the group element that results from the product $g_{i}g_{j}$.
## 1.2 Dihedral Groups
N/A
## 1.3 Symmetric Groups
N/A
## 1.4 Matrix Groups
**Definition:**
- A *field* is a set $F$ together with two commutative binary operations $+$ and $\cdot$ on $F$ such that $(F,+)$ is an abelian group (call its identity $0$) and $(F-\{ 0 \},\cdot)$ is also an abelian group, and the following *distributive law* holds: $a\cdot(b+c)=(a\cdot b)+(a\cdot c)$, $\forall a,b,c\in F$.
- For any field $F$ let $F^{\times}=F-\{ 0 \}$.

## 1.5 Quaternion Group
N/A
## 1.6 Homomorphisms and Isomorphisms
**Definition:** Let $(G,*)$ and $(H,\circ)$ be groups. A map $\varphi:G\to H$ such that $\varphi(x*y)=\varphi(x)\circ \varphi(y)$, $\forall x,y\in G$, is called a *homomorphism*.
**Definition:** The map $\varphi:G\to H$ is called an *isomorphism* and $G$ and $H$ are said to be *isomorphic* or of the same *isomorphism type*, written $G\cong H$, if
- $\varphi$ is a homomorphism (i.e. $\varphi(xy)=\varphi(x)\varphi(y)$)
- $\varphi$ is a bijection.

**Fact:** $\varphi$ is an isomorphism $\iff$ there exists a well defined inverse homomorphism $\varphi ^{-1}$.
**Fact:** If $\varphi$ is an isomorphism, then: $\lvert G \rvert=\lvert H \rvert$, $G$ is abelian $\iff H$ is abelian, $\forall x \in G$ it is true $\lvert x \rvert=\lvert \varphi(x) \rvert$.
## 1.6, 1.7 Homomorphisms, Group Actions
**Definition:**  A *group action* of a group $G$ on a set $A$ is a map from $G\times A\to A$ satisfying
- $g_{1}(g_{2}a)=(g_{1}g_{2})a,\ \forall g_{1},g_{2}\in G$ and $\forall a\in A$
- $1\cdot a=a$, $\forall a\in A$.

**Fact:** For all fixed $g\in G$, $\exists\sigma_{g}$, a permutation of $A$ where the map $g\mapsto\sigma_{a}$ is a homomorphism.
<div style="page-break-after: always;"></div>