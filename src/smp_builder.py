import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps


class SMPBuilder(object):
    """Simple class that creates a synthetic SMP using input from the command line."""

    def __init__(self, model_parameters, start_time, vector_length):
        self.alpha, self.beta, self.a_0 = model_parameters
        self.t_0 = start_time
        self.capture_len = vector_length
        self.pulse_len = self.capture_len - self.t_0
        self._generate_baseline()
        self._generate_pulse()

    def _generate_pulse(self):
        """Generates a gamma distribution based on input variables and then places
           it at the requested spot in a synthetic timeline."""
        shape, scale = self.alpha, 1 / self.beta
        x_vals = np.linspace(.00001, 1.3, self.pulse_len)
        self.pulse = (x_vals ** (shape - 1) * (np.exp(-x_vals / scale)) /
                      (sps.gamma(shape) * scale ** shape))
        scale_factor = np.max(self.pulse)
        self.pulse = self.pulse / scale_factor
        self.synth_smp = self.baseline
        self.synth_smp[self.t_0:] = self.synth_smp[self.t_0:] + self.pulse

    def _generate_baseline(self, start_mean=0.06, start_std=0.077):
        """Generates a base timeline with proper length, appropriate mean and standard deviation."""
        # assume counts are randomly distributed with small mean and standard deviation
        self.baseline = (np.random.random(self.capture_len) * start_std) + start_mean

    def write_csv(self, csv_name):
        """Writes the synthetic SMP to a CSV."""
        # write it all out as a CSV.
        with open(csv_name, 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            py_floats = self.synth_smp.tolist()
            for time, y_val in enumerate(py_floats):
                csv_writer.writerow([time, y_val])


def get_arguments():
    """Commandline argument parser that takes arguments in the following order:
        'start_time vector_length alpha beta a_0 output_csv.csv'"""
    parser = argparse.ArgumentParser(description='Generate synthetic SMP data.')
    parser.add_argument('start_time', metavar='start_time', type=int, \
                        help='choose when the pulse should start, in samples for now')
    parser.add_argument('vector_length', metavar='vector_length', type=int, \
                        help='how long should the total vector be? (in samples for now)')
    parser.add_argument('model_parameters', metavar='model_pars', type=float, nargs='+', \
                        help='SMP parameters in the order alpha, beta, a0')
    parser.add_argument('out_csv', help='filename for output CSV')

    args = parser.parse_args()
    return args.model_parameters, args.start_time, args.vector_length, args.out_csv

def main():
    """Generates a synthetic SMP, plots it, and writes it to CSV."""
    model_params, start_time, vector_length, out_csv = get_arguments()
    smp = SMPBuilder(model_params, start_time, vector_length)

    # plot what you've come up with
    plt.plot(smp.synth_smp)
    plt.show()

    smp.write_csv(out_csv)


if __name__ == "__main__":
    main()
