The governing equation for 1D steady flow is

$$
\begin{align}
\frac{a_{x x}-1}{W i}\left[\epsilon\left(a_{x x}-1\right)+1\right]-\kappa a_{x x}^{\prime \prime}=2 a_{x y} U_{l a m}^{\prime}, \\
\frac{a_{x y}}{W i}\left[\epsilon\left(a_{x x}-1\right)+1\right]-\kappa a_{x y}^{\prime \prime}=U_{l a m}^{\prime}, \\
\beta U_{l a m}^{\prime \prime}+\frac{1-\beta}{W i} a_{x y}^{\prime}+2=0 .
\end{align}
$$
and the solution is $\boldsymbol{v}_{l a m}=\left(U_{l a m}(y), 0,0\right)$ and
$$
\begin{equation}
\boldsymbol{c}_{l a m}=\left(\begin{array}{cc}
a_{x x}(y) & a_{x y}(y) \\
a_{x y}(y) & 1
\end{array}\right)
\end{equation}
$$
----
## Solution with $\kappa=0$
We now want to solve
$$
\begin{align}
\frac{a_{x x}-1}{W i}\left[\epsilon\left(a_{x x}-1\right)+1\right]=2 a_{x y} U_{l a m}^{\prime}, \\
\frac{a_{x y}}{W i}\left[\epsilon\left(a_{x x}-1\right)+1\right]=U_{l a m}^{\prime}, \\
\beta U_{l a m}^{\prime \prime}+\frac{1-\beta}{W i} a_{x y}^{\prime}+2=0 .
\end{align}
$$
We can define the total shear stress as
$$
\begin{equation}
    \tau(y) = \beta U_{l a m}^{\prime}+\frac{1-\beta}{W i} a_{x y}
\end{equation}
$$
and the symmetry in channel geometry requires that
$$
\begin{equation}
    \tau_{w}=\tau(-1) = -\tau(+1).
\end{equation}
$$
If we integrate equation (7) from $y=-1$ to $+1$, we have
$$
\begin{equation}
    \tau_w = 2.
\end{equation}
$$
Therefore, the momentum equation (7) can be reduced by one degree as
$$
\begin{equation}
    \tau(y) = \beta U_{l a m}^{\prime}+\frac{1-\beta}{W i} a_{x y} = -2y,
\end{equation}
$$
and we can solve equation (5), (6) and (11) for $U_{l a m}^{\prime}$, $a_{x x}$, $a_{yy}$ and $a_{x y}$ in terms of $y$ and fluid parameters. It is easy to obtain
$$
\begin{equation}
    a_{xx}-1= 2a_{xy}^2
\end{equation}
$$
by inspecting equation (5) and (6). Then a cubic equation for $a_{xy}$ can be fetched from equation (6) and (11)
$$
\begin{equation}
    a_{xy}^3+\frac{1}{2\beta\epsilon}a_{xy} + \frac{Wi}{\beta \epsilon}y=0
\end{equation}
$$
Let $A=1/(6\beta\epsilon)$ and $B=Wi/(2\beta\epsilon)$, we have
$$
\begin{equation}
    a_{xy}^3+3Aa_{xy} + 2By=0
\end{equation}
$$
We have the solution
$$
a_{xy}=F^{+}(y) + F^{-}(y)
$$
where
$$
F^{\pm}(y) = \sqrt[3]{-By\pm \sqrt{A^3 + (By)^2}}
$$

---
# References
Morozov, A. (2022). Coherent Structures in Plane Channel Flow of Dilute Polymer Solutions with Vanishing Inertia. Physical Review Letters, 129(1), 017801. https://doi.org/10.1103/PhysRevLett.129.017801
