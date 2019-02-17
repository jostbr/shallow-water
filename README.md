# Shallow water equations
Model solving the 2D [shallow water equations](https://en.wikipedia.org/wiki/Shallow_water_equations). In particular, the momentum equations are linearized while the continuity equation is solved non-linearly. The model was developed as part of the ["Bornö Summer School in Ocean Dynamics"](https://chess.w.uib.no/event/borno-summer-school-practice/) and in order to study geophysical fluid dynamics theory evolve in a numerical simulation.

As it is set up right now, the model initates with a large gaussian bump resulting in waves propagate away from the bump, then interacts with the walls (no flow condition). The solution for the surface elevation can be seen below

![Surface elevation solution](surface.gif)

and the associated flow field looks like

![Velocity field solution](velocity.gif)

Feel free to play around with the parameters to your liking. You might have to make some tweeks to the colorscale and arrowscale for some of the plots if you change some parameters.
