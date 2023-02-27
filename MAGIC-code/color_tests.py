import matplotlib.colors as mc
import colorsys

def lighten_color(color, amount):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    referance: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    adjusted_c =  colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    return '#%02x%02x%02x' % (int(adjusted_c[0] * 255), int(adjusted_c[1] * 255), int(adjusted_c[2] * 255))

#####
color = 'blue'

print("source:", color)

adjusted = lighten_color(color, 0.75)
print("75%:", adjusted)

adjusted = lighten_color(color, 0.5)
print("50%:", adjusted)

adjusted = lighten_color(color, 0.25)
print("25:%", adjusted)