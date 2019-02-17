# Shallow water equations
Model solving the 2D [shallow water equations](https://en.wikipedia.org/wiki/Shallow_water_equations). The momentum equations are linearized while the continuity equation is solved non-linearly. The model was developed as part of the ["Bornö Summer School in Ocean Dynamics"](https://chess.w.uib.no/event/borno-summer-school-practice/) partly to study theory evolve in a numerical simulation. Visualizations scripts are also provided.

As it is set up right now, the model initates with a large gaussian bump resulting in waves propagate away from the bump, then interacts with the walls (no flow condition). The solution for the surface elevation and velocity fields can be seen below:

<div class="nav3" style="height:705px;">
    <img src="surface.gif" alt="Surface elevation solution" width="47%"></a>
    <img src="velocity.gif" alt="Velocity field solution" width="47%"></a>
</div>

Feel free to play around with the parameters to your liking. You might have to make some tweeks to the colorscale and arrowscale for some of the plots if you change some parameters.
