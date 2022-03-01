# Introduction

`Leo.py` is python script that deals with low Earth orbit (LEO) constellations, does numeric simulations, and output data for visualization (via Cesium by AGI) and NDN simulation (via ndnSIM 2.x).

The script implements the following major functions with configurable parameters and alternative algorithms:
- generate constellations: parameters include orbit height, inclination, number of orbit planes, number of satellites per plane, and orbit plane distribution pattern (RAAN)
- generate ground stations: at major cities or given geographic coordinates
- generate inter-satellite topology by setting ISLs: a simple grid-like setup is implemented
- generate satellite attachments (which satellite a ground station connects to) at each epoch: three handover strategies are provided
- compute routes between pairs of ground stations: a simple shortest-path route selection criteria is implemented
- compute the intersection of paths before and after satellite handover for each pair of ground station: the intersection point closest to the consumer's current location is selected

The script also generates outputs for external uses:
- generate CZML file for visualization via Cesium
- generate input file for ndnSIM simulations

# Dependencies

`Leo.py` is a **Python 3.x** script and depends on the following 3rd party libraries:

`ephem skyfield czml networkx tqdm`