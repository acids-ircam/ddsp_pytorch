# Dataspaces and Units of Measurement

Dataspaces for attributes are an out-growth from the Jamoma project and the paper at NIME or ICMC or somewhere...

## Spatial Units

I'm sure that if you review the code for this dataspace you'll find oddities several places. The reason is because we are using non-standard mathematical conversions:

- Cartesian coordinates: x to right, y forward, z upwards
- Polar: Navigational system with 0 degrees to the north (forward), +90 to the east (right), -90 to the west (left) and 180 south (back).

This makes the formulas for conversions between Cartesian and Polar different to what you'd expect in a maths class.

