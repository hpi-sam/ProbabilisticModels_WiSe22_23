import ast
import numpy as np
import pandas as pd
import scipy


def main():
    profile_results = pd.read_csv(
        '../p3/profile.csv',
        index_col=False,
        converters={"state": ast.literal_eval, "next_state": ast.literal_eval}
    )

    profile_results = profile_results.sort_values(by=['state', 'next_state'])
    profile_results = profile_results.drop(columns=['episode'])

    profile_results = pd.DataFrame((profile_results.groupby(['state', 'next_state']).apply(
        lambda x: fit(x['time'])
    )))
    profile_results['λ'], profile_results['stddev'] = zip(*profile_results[0])
    profile_results = profile_results.drop(columns=[0])

    #print(profile_results)

    visualize(profile_results)

    # TODO NEXT: add remaining states as NaN, use code from p4_1 to compute limiting distribution. should we set a high epsilon in p3 for covering more states?


def fit(times):
    X = np.sort(times)
    Y = np.array([len(times[times <= x]) / len(times) for x in X])

    (λ,), x = scipy.optimize.curve_fit(lambda t, λ: 1 - np.exp(-λ * t), X, Y, p0=(1,))
    stddev = np.sqrt(np.diag(x))[0]

    return λ, stddev




def visualize(profile_results):
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    # draw the cliffmaze image
    width, height = max([t for [s, t] in profile_results.index.values])
    width += 1
    height += 1
    img = Image.new('RGB', (width * 100, (height + len(['legend'])) * 100), color='white')
    draw = ImageDraw.Draw(img, 'RGBA')
    for x in range(width):
        for y in range(height):
            draw.rectangle([(x * 100, y * 100), ((x + 1) * 100, (y + 1) * 100)], fill='white')
    for x in range(width):
        draw.line([(x * 100, 0), (x * 100, height * 100)], fill='black')
    for y in range(height):
        draw.line([(0, y * 100), (width * 100, y * 100)], fill='black')
    draw.rectangle([(0, 0), (width * 100, height * 100)], outline='black')
    draw.rectangle([(0, height * 100 - 100), (100, height * 100)], fill='blue')
    draw.rectangle([(width * 100 - 100, height * 100 - 100), (width * 100, height * 100)], fill='blue')
    # cliff
    for x in range(1, width - 1):
        draw.rectangle([(x * 100, height * 100 - 100), ((x + 1) * 100, height * 100)], fill='grey')

    # colorize the arrows between the states according to the λ values (mix red for the smallest value, green for the largest value)
    for (s, t), (λ, stddev) in profile_results[['λ', 'stddev']].iterrows():
        relative_value = (λ - profile_results['λ'].min()) / (profile_results['λ'].max() - profile_results['λ'].min())
        relative_confidence = 1 - (stddev - profile_results['stddev'].min()) / (profile_results['stddev'].max() - profile_results['stddev'].min())

        color = tuple([int(255 * (1 - relative_value)), int(255 * relative_value), 0, int(255 * relative_confidence)])

        # fix coordinates when jumping back to start (looks cleaner)
        if t == (0, height - 1):
            t = [s[0], height - 1]

        size = 0.125 + (0.5 - 0.125) * relative_confidence
        if t[0] > s[0]:
            # right
            draw.polygon([(t[0] * 100, (t[1] + 0.5 - size) * 100), (t[0] * 100, (t[1] + 0.5 + size) * 100), ((t[0] + size) * 100, (t[1] + 0.5) * 100)], fill=color)
        elif t[1] > s[1]:
            # bottom
            draw.polygon([((t[0] + 0.5 - size) * 100, t[1] * 100), ((t[0] + 0.5 + size) * 100, t[1] * 100), ((t[0] + 0.5) * 100, (t[1] + size) * 100)], fill=color)
        elif t[0] < s[0]:
            # left
            draw.polygon([(t[0] * 100, (t[1] + 0.5 - size) * 100), (t[0] * 100, (t[1] + 0.5 + size) * 100), ((t[0] - size) * 100, (t[1] + 0.5) * 100)], fill=color)
        elif t[1] < s[1]:
            # top
            draw.polygon([((t[0] + 0.5 - size) * 100, (t[1] + 1) * 100), ((t[0] + 0.5 + size) * 100, (t[1] + 1) * 100), ((t[0] + 0.5) * 100, (t[1] + 1 - size) * 100)], fill=color)
        # could refactor, but could also not

    # legend
    draw.rectangle([(0, height * 100), (width * 100, height * 100 + 100)], fill='white')
    draw.text((0, height * 100 + 10), f"rate: green = high ({profile_results['λ'].max():.2f}), red = low ({profile_results['λ'].min():.2f})", fill='black')
    draw.text((0, height * 100 + 30), f"stddev: small/translucent = low ({profile_results['stddev'].min():.2f}), large/opaque = high ({profile_results['stddev'].max():.2f})", fill='black')

    # save the image
    img.save('p4_2.png')


if __name__ == '__main__':
    main()
