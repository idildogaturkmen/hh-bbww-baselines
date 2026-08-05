def extract_full_histogram_data(scores, bins):
    """
    Extract data for the full symmetric histogram from triangular scores
    """
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    n_bins = len(bin_centers)

    # Create empty histogram
    H = np.zeros((n_bins, n_bins))

    # Fill triangular region
    idx = 0
    for i in range(n_bins):
        for j in range(i, n_bins):
            H[i, j] = scores[idx]
            idx += 1

    # Create symmetric histogram
    H_sym = (H + H.T) / 2

    return H_sym, bin_centers
