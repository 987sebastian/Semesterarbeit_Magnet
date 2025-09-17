from scripts.Geometry import CylinderGeom
from scripts.Discretization import DiscGrid
from scripts.generate_cpp import write_autogen

if __name__ == "__main__":
    write_autogen(CylinderGeom(Br=1.2, R=0.005, l=0.004), DiscGrid(), outdir="..")