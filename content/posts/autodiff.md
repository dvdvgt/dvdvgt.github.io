+++
title = "Forward Automatic Differentiation"
date = "2025-02-26"
+++

Recently, I was fascinated by the ideas presented in the wounderfull [video](https://www.youtube.com/watch?v=QwFLA5TrviI) by [Computerphile](https://www.youtube.com/@Computerphile) on YouTube.
It is about automatically computing the derivatives.
I found the idea and concept very interesting and tried to re-iterate them in more detail and more formally here.

From your high school calculus days, you likely recall symbolic differentiation—though perhaps not by that specific name.
When faced with a function like $f(x)=sin⁡(x)^2$, you'd methodically apply the chain rule to derive $f'(x)=2 \sin⁡{x} \cos⁡{x}$.
This process relies on memorizing fundamental identities such as $\frac{\mathrm{d}}{x} \ln{x} = \frac{1}{x}$, the chain rule $\frac{\mathrm{d}}{x} f(g(x)) = f'(g(x)) g'(x)$, or the product rule.

When computing derivatives, two primary approaches have traditionally dominated the field:

+ **Symbolic differentiation**: Deriving analytical forms by systematically applying mathematical identities until reaching a closed-form solution.
+ **Numerical differentiation**: Employing numerical algorithms to approximate derivatives.

While symbolic differentiation offers intuitive appeal and works elegantly on paper for simpler expressions, automating this process comes with significant challenges. The function must be parsed into an abstract syntax tree that captures the precise order of operations. Although this approach yields exact results, it often generates redundant calculations due to the chain and product rules, creating serious computational inefficiencies for complex expressions.
Numerical methods, while potentially faster and more flexible, sacrifice precision and can introduce stability issues that undermine their reliability.

There is another approach that combines the benefits of both the other approaches and while mostly eliminating their respective disadvantages: **automatic differentiation**.
If you work in deep learning, you're already benefiting from automatic differentiation through backpropagation—which is actually reverse-mode automatic differentiation. This algorithm efficiently computes gradients through complex neural networks, making modern deep learning computationally feasible. Without this approach, training sophisticated models would remain prohibitively expensive.
In this post, I will introduce the foundational concepts of automatic differentiation and present an elegant implementation of forward automatic differentiation in Haskell, revealing the mathematical beauty and computational efficiency that make this technique indispensable in modern computational science.

## Automatic Differentiation

The chain rule is the fundamental concept for automatic differentiation.
Consider the composite function $h(x) = f(g(d(x)))$ with $u_3 = y = f(u_2), u_2 = g(u_1), u_1 = l(u_0), u_0 = x$, then the chain rules tells us that
$$
\frac{\partial y}{\partial x} = \frac{\partial f(u_2)}{\partial u_2} \frac{\partial g(u_1)}{\partial u_1} \frac{\partial l(x)}{\partial x} = \frac{\partial y}{\partial u_2} \frac{\partial u_2}{\partial u_1} \frac{\partial u_1}{\partial x}
$$
This yields two directions in which we can calculate the overall derivative:
1. **Forward mode**: We start with the right-most term and then recursively work our way "outwards". That is, we recursively compute
   $$
   \frac{\partial u_i}{\partial x} = \frac{\partial u_i}{\partial u_{i - 1}} \frac{\partial u_{i - 1}}{\partial x}
   $$
   and start with $\frac{\partial u_0}{\partial x} = \frac{\partial x}{\partial x} = 1$
2. **Backward (reverse) mode**: We start with the right-most term and then recursively work our way "inwards":
   $$
   \frac{\partial y}{\partial u_i} = \frac{\partial y}{\partial u_{i + 1}} \frac{\partial u_{i + 1}}{\partial u_i}
   $$
   and we start with $\frac{\partial y}{\partial u_n} = \frac{\partial y}{\partial y} = 1$.

Notice how the forward mode keeps the independent variable fixed and thereby computes the derivative for each variable in one separate pass.
Reverse mode, on the other hand, requires the evaluated partial functions for the partial derivatives. Thus, function is evaluated first and then the derivatives with respect to all independent variables is calculated in an additional pass.

## Dual Numbers

Now, consider a function $f(x)$ and the taylor expansion of $f$ around some $\epsilon$ such that $\epsilon^2 = 0$ and $\epsilon \neq 0$:

$$
\begin{align*}
f(x + \epsilon) &= \sum_{d = 0}^\infty \frac{f^d(x)}{d!} \epsilon^d \\\\
&= f(x) + f'(x) \cdot \epsilon + f''(x) \cdot \epsilon^2 + f'''(x) \cdot \epsilon \epsilon^2 + \dots \\\\
&= f(x) + f'(x) \epsilon
\end{align*}
$$

Notice that the factor of $\epsilon$ corresponds to the derivative of $f$.
What happens if we have another function $g$ and we try to add or multiply it?

$$
\begin{align*}
f(x + \epsilon) + g(x + \epsilon) &= f(x) + f'(x) \epsilon + g(x) + g'(x) \epsilon = f(x) + g(x) + (f'(x) + g'(x)) \epsilon \\\\
f(x + \epsilon) \cdot g(x + \epsilon) &= (f(x) + f'(x)\epsilon) \cdot (g(x) + g'(x) \epsilon) = f(x) g(x) + (f(x) g'(x) + g(x) f'(x)) \epsilon
\end{align*}
$$

You might notice these as the sum and product rule of derivatives!
If we squint a bit, we see that the first term always corresponds to the function we want to compute the derivative of and the second term represents the actual derivative.
We ca leverage this for automatically computing the forward derivative of arbitrary function for which a taylor expansion exists.

We define dual numbers as a pair
$$d = \langle a, b \rangle$$
where $a = f(x)$ and $b = f'(x)$ (of the $\epsilon$ term).
For example, we can write $\sin(x)$ as the dual number
$$\langle \sin(x), \cos(x) \rangle$$
or 
$$\langle \ln x, \frac{1}{x} \rangle, \quad \langle e^x, e^x \rangle$$
Using the sum, product rule and division rule, that is,
$$
\begin{align*}
\langle a, b \rangle + \langle c, d \rangle &= \langle a + d, b + c \rangle \\\\
\langle a, b \rangle \cdot \langle c, d \rangle &= \langle a \cdot b, ad + cb \rangle \\\\
\langle a, b \rangle / \langle c, d \rangle &= \langle a / c, \frac{bc - ad}{c^2} \rangle
\end{align*}
$$
we can piece together the derivative for more complex functions.
Consider, for example, the sigmoid (activation) function $\sigma(x) = \frac{e^x}{1 + e^x}$ at some $x$ represented as the dual number $\langle x, 1 \rangle$:
$$
\begin{align*}
\frac{e^{\langle x, 1 \rangle}}{\langle 1, 0 \rangle + e^{\langle x, 1 \rangle}} &= \frac{\langle e^x, e^x \rangle}{\langle 1+  e^x, e^x \rangle} \\\\
&= \frac{\langle e^x, e^x \rangle}{\langle 1 + e^x, e^x \rangle} \\\\
&= \langle \frac{e^x}{1 + e^x}, \frac{e^x (1 + e^x) - e^{2x}}{(1 + e^x)^2} \rangle \\\\
&= \langle \frac{e^x}{1 + e^x}, \frac{1}{1 + e^x} \cdot \frac{e^x}{1 + e^x} \rangle \\\\
&= \langle \sigma(x), (1 - \sigma(x)) \sigma(x) \rangle
\end{align*}
$$
Notice that we thread through the input and arrive at the valid derivative automatically.

## Implementation

For the implementation, I chose Haskell due to its nice overloading capabilities via type classes.

First, we define a datatype for dual numbers:

```haskell
data Dual = { real :: Double, dual :: Double }
```

Next, we implement the `Num` and `Fractional` type class since we want to overload the `+`, `-`, `*` and `/` operator:

```haskell
instance Num Dual where
  (Dual a b) + (Dual c d) = Dual (a + c) (b + d)
  (Dual a b) * (Dual c d) = Dual (a * c) (a * d + b * c)
  negate (Dual a b) = Dual (negate a) (negate b)
  fromInteger n = Dual (fromInteger n) 0
  -- abs and signum are not defined for Dual numbers
  abs = undefined
  signum = undefined

instance Fractional Dual where
  (Dual a b) / (Dual c d) = Dual (a / c) ((b * c - a * d) / (c * c))
  fromRational r = Dual (fromRational r) 0
```

Notice that these exactly correspond to the rules we previously derived formally.

Of course automatic differentiation is not pure magic and need to define the derivatives of common functions once manually.
This can be done by implenting the `Floating` type class where many commonly used mathematical functions are defined and can be overloaded to also accept dual numbers:

```haskell
instance Floating Dual where
  pi = Dual pi 0
  exp (Dual a b) = Dual (exp a) (b * exp a)
  log (Dual a b) = Dual (log a) (b / a)
  sin (Dual a b) = Dual (sin a) (b * cos a)
  cos (Dual a b) = Dual (cos a) (-b * sin a)
  asin (Dual a b) = Dual (asin a) (b / sqrt (1 - a * a))
  acos (Dual a b) = Dual (acos a) (-b / sqrt (1 - a * a))
  atan (Dual a b) = Dual (atan a) (b / (1 + a * a))
  sinh (Dual a b) = Dual (sinh a) (b * cosh a)
  cosh (Dual a b) = Dual (cosh a) (b * sinh a)
  asinh (Dual a b) = Dual (asinh a) (b / sqrt (1 + a * a))
  acosh (Dual a b) = Dual (acosh a) (b / sqrt (a * a - 1))
  atanh (Dual a b) = Dual (atanh a) (b / (1 - a * a))
```

Now we get to the actual interesting part: automatically compute the derivative for any function of the type $\mathbb{R} \to \mathbb{R}$.
Hence, we define the function `diff` with the type `(Dual -> Dual) -> Double -> Double`, that is, a function that accepts a function `f` and an argument `x` at which the derivative shall be evaluated:

```haskell
diff :: (Dual -> Dual) -> Double -> Double
diff f x = dual $ f (Dual x 1)
```

Magical, isn't it?
Now we have an embedded DSL for computing derivatives in Haskell:

```haskell
main = do
  let f x = 1 / (1 + e ** (- x))
  putStrLn $ diff f 0
```

Alternatively, you can also simply start a REPL session using `ghci`:

```bash
$ ghci dual.hs
ghci> f x = 1 / (1 + e ** (- x))
ghci> diff f 0
```

Running this yields `0.25` as result.
Plugging this into the analytical form we have found previously yields the same result:
$$\frac{e^x}{1 + e^x} \frac{1}{1 + e^x} = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4}$$

## Extensions

I hope you also found this as fascinating as I did when I first heard about it.
There are some interesting generalisation we can apply to our current implementation as an exercise:

1. Think about how to compute the partial derivative of functions $\mathbb{R} \to \mathbb{R}^m$ and implement a function `partial :: (Dual -> [Dual]) -> Int -> Double -> [Double]` which computes the partial derivatives with respect to the i-th parameter.
2. How can you generalise `partial` to also accept functions $\mathbb{R}^n \to \mathbb{R}^m$? Adapt the implementation of `partial` such that it has the type `([Dual] -> [Dual]) -> Int -> Double -> [Double]`.
3. Finally, implement a function `gradient` that computes the gradient of functions $\mathbb{R}^n \to \mathbb{R}^m$. `gradient` now shall have the type `([Dual] -> [Dual]) -> Int -> Double -> [[Double]]`, that is, it returns a matrix. Hint: `partial` should come in handy here. What do you observe? When is forward AD to be preferred? If $n < m$ or $n > m$?