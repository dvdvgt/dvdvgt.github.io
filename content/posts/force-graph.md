+++
title = "Force-Directed Graph Drawing"
date = "2026-02-08"
draft = false
description = ""
template = "article.html"
+++

Graphs offer a nice theoretical framework for elegantly describing and solving many practical problems like shortest-path (Dijkstra, A*, BFS, DFS), network routing (max-flow) or dependency resolution (topological ordering).
Since graph are such a general concept and surface in wide variety of problem domains, the visualisation of graphs can be highly insightful.
Yet, graph drawing, the combination of geometry with graph theory, can be difficult problem.

Recently, I attended a talk in which the live-demo featured such a visualised graph.
I found the result very aesthetically pleasing and enjoyed the interactiveness of it. 
You could freely drag and move nodes and the graph would re-arrange itself based on the new initial configuration in a nice way.
I was intruiged and decided to implement such a force-direct graph drawing (FDGD) algorithm myself.
Here is what I learned on the way.

![](/fdgd.png)

## Basics

Theoretically, a graph $G$ can be relatively easily defined.
A graph $G = (V, E)$ is tuple consisting of a set of vertices/nodes $V$ and a set of edges connecting the vertices.
More formally

$$V := \\{ v_1, v_2, \dots, v_n \\}, \quad E := \\{ \\{u, v\\} \mid u, v \in V \\}.$$

Depending on whether the direction of the edges is of importance, the graph is either considered directed or undirected.
In the following, we will just consider undirected graphs for simplicity.

Now, based on this definition, we are completely free to place each vertex whereever we want given that we connected all vertices in $V$ by the edges defined by $E$.
However, of course, not all possible configurations are aesthetically pleasing or even instructive.
There are a lot of different ways to measure the quality of a graph visualisation like the number of crossing edges or minimising the maximum length of all edges.
Convienently, FDGD inherently checks many of these boxes, so lets see how it works.

## The Force Will Be with You, Always

Classical FDGD algorithms achieve this by using only two forces: an attracting force and a repulsive force. 
The repulsive force is often derived from [Coulomb's Law](https://en.wikipedia.org/wiki/Coulomb%27s_law):

$$F_r = k_r \frac{|q_1| |q_2|}{r^2}$$

Like electrons are repulsing one another inversely proportional to the distance between them, each vertex in a graph repulses all other vertices:
Since our vertices do not have a charge or rather have all the same charge, we can just subsume this by the constant $k_e$.
Whereas $k_r$ is originally the Coulomb's Constant, we are free to choose our own constant here.

Having now covered repulsive force, we are still missing an attracting force that binds the graph together.
For this, we choose [Hooke's Law](https://en.wikipedia.org/wiki/Hooke%27s_law) which models the spring-force induced by the graph's edges.

$$F_a = r \cdot k_a$$

That is, if we think about the edges as springs, the amount of force needed to further extend the spring sclaes linearly with the length of the extension $r$ and some constant $k_a$.

## Implementation

That is it. This is basically all we need to nicely draw graphs.
We have the repulsive force

```js
repel(other: Vertex, params: Parameters) {
    const dx = other.x - this.x
    const dy = other.y - this.y
    const dist = Math.hypot(dx, dy) || 0.01

    const force = params.repulsion / (dist * dist)
    const fx = (force * dx) / dist
    const fy = (force * dy) / dist

    this.apply(-fx, -fy)
    other.apply(fx, fy)
}
``` 

and the attracting force:

```js
attract(other: Vertex, params: Parameters) {
    const dx = other.x - this.x
    const dy = other.y - this.y
    const dist = Math.hypot(dx, dy) || 0.01

    const force = (dist - params.springLength) * params.springStrength
    const fx = (force * dx) / dist
    const fy = (force * dy) / dist

    this.apply(fx, fy)
    other.apply(-fx, -fy)
}

apply(fx: number, fy: number) {
    this.vx += fx
    this.vy += fy
}
```

Finally, we also apply some damping and limit the speed:

```js
integrate(params: Parameters) {
    const maxv = params.maxVelocity
    this.vx = Math.max(Math.min(this.vx, maxv), -maxv)
    this.vy = Math.max(Math.min(this.vy, maxv), -maxv)
    this.vx *= params.damping
    this.vy *= params.damping
    this.x += this.vx
    this.y += this.vy
}
```

Notice that we are accumulating forces and then in the end just add them to the velocity.
This is of course a _very_ simplified physical model.
Normally, force is $kg \cdot \frac{m}{s^2}$, the velocity is the change in position $\frac{dx}{dt} = v(t)$, and the accelerations is in turn the change in velocity $\frac{dv}{dt} = a(t)$.
We can, however, try to justify our implementation by assuming that the mass is just one ($m = 1 kg$), and since we are only considering discreet time steps, $dt = 1$. 
Under these assumption, we can derive the following (simplified) update equations for the position and velocity:

$$
\begin{align*}
x(t + 1) - x(t) &= v(t) \\\\
\Leftrightarrow x(t + 1) &= x(t) + v(t) \\\\
v(t + 1) - v(t) &= a(t) \\\\
\Leftrightarrow v(t + 1) &= v(t) + a(t) \\\\
&= v(t) + F(t)
\end{align*}
$$

I invite you to check out the implementation at <https://fdgd.dvoigt.de>.
