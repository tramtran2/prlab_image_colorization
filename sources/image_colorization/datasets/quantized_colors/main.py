import pprint
import click
from utils import cielab_color_space, compute_prior_prob_export, compute_prior_prob_smoothed, compute_prior_factor

@click.group(invoke_without_command=False)
def main():
    print("---------------------------------------")
    print("CALCULATE QUANTIZED COLOR INDEX")
    print("---------------------------------------")
# main

@main.command()
def main_cielab_color_space(**input_params):
    print("---------------------------------------")
    print("PRINT COLOR RANGE OF SKLEARN AND CV2")
    print("---------------------------------------")
    cielab_color_space()
# main_cielab_color_space

@main.command()
@click.option("--db_root", type=str)
@click.option("--db_file", type=str)
@click.option("--db_name", type=str)
@click.option("--column_image", type=str, default = "image")
@click.option("--column_type", type=str, default = "type")
@click.option("--process_types", multiple=True, type=str, default = 'train')
@click.option("--pts_in_hull_path", type=str)
@click.option("--export_prior_prob_path", type=str)
@click.option("--export_ab_hist_path", type=str)
@click.option("--is_resize", type=bool, default = False)
@click.option("--width", type=int)
@click.option("--height", type=int)
@click.option("--do_plot", type=bool, default = False)
@click.option("--verbose", type=bool, default = False)
def main_compute_prior_prob(**input_params):
    print("---------------------------------------")
    print("EXPORT QUANTIZED COLOR INDEX FOR DATASET")
    print("---------------------------------------")
    print(pprint.pformat(input_params))
    print("---------------------------------------")
    compute_prior_prob_export(**input_params)
    print("---------------------------------------")
# main_compute_prior_prob

@main.command()
@click.option("--prior_prob_path", type=str)
@click.option("--prior_prob_smoothed_path", type=str)
@click.option("--sigma", type=float, default = 5.0)
@click.option("--do_plot", is_flag=True)
@click.option("--verbose", type=int, default = 1)
def main_compute_prior_prob_smoothed(**input_params):
    print("---------------------------------------")
    print("SMOOTH PRIOR PROBABILITY")
    print("---------------------------------------")
    print(pprint.pformat(input_params))
    print("---------------------------------------")
    compute_prior_prob_smoothed(**input_params)
    print("---------------------------------------")
# main_compute_prior_prob_smoothed

@main.command()
@click.option("--prior_prob_path", type=str)
@click.option("--prior_prob_smoothed_path", type=str)
@click.option("--prior_prob_factor_path", type=str)
@click.option("--gamma", type=float, default = 0.5)
@click.option("--alpha", type=float, default = 1.0)
@click.option("--do_plot", is_flag=True)
@click.option("--verbose", type=int, default = 1)
def main_compute_prior_factor(**input_params):
    print("---------------------------------------")
    print("FACTORIZE PRIOR PROBABILITY SMOOTHED")
    print("---------------------------------------")
    print(pprint.pformat(input_params))
    print("---------------------------------------")
    compute_prior_factor(**input_params)
    print("---------------------------------------")
# main_compute_prior_factor

if __name__=="__main__":
    main()
# if
