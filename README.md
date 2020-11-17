# Running the code
This code is used to produce the results of https://arxiv.org/abs/2004.10292.
The code is run by `python3 main.py k` where `k` represents an experiment
number. For example `python3 main.py 0`gerenates the first table in
`tables/2D_Hartmann-P2_P1_P1.txt`. More on this later. The experiments are
generated in pickle files contained in the pickles/ directory. Creating new
pickle files will be discussed in the next section.

# Changing parameters with pickler.py
All the parameters of the code can be changed through `pickler.py`. In
particular, the workflow is to change the desired parameter(s) in `pickler.py`,
run the command ```python3 main.py pickle k``` where `k` is the number of the
new experiment (currently there are 0-7 already in use), and then `python3
main.py k` will run your experiment with the updated parameters.

# Tables, plotting and XMLs
The code by default writes the LaTeX tables to the `tables/` directory. The
files therein are organized by first the number of dimensions (currently 2D for
all experiments in the article) then experiement name (Hartmann or lid_driven)
then the polynomial order for the primal product space e.g. `P2_P1_P1` for (u,
p, B). The tables are printed out in LaTeX table format so they can be easily
copied and pasted. You can toggle plotting by the `do_plotting` flag in
`pickler.py`. Finally, for the lid driven cavity, the solution automatically
gets saved in the XMLs directory. Then, the next time the specific parameters
are used, the saved file is read in as an inital guess for the nonlinear
solver. This allows for the homotopy argument for the Reynold's number.

# Pickle files 4-7 for the lid driven cavity
For these experiments, an initial guess is being loaded from the files in the
folder `init_guesses` which can be extracted `tar -xvf init_guesses.tar`.  This
is necessary to run the lid driven cavity tests, since the solver stalls for
large Reynold's number.

# Running in parallel
Thanks to the magic of FEniCS, one can seamlessly run the code in parallel by
using `mpiexec -n <np> python3 main.py k`.

# Collaborators
Ari Rappaport  - Inria, Paris\
Jehanzeb Chaudhry - University of New Mexico\
John Shadid - University of New Mexico/Sandia National Laboratories\
J. Chaudhry’s work is supported by the NSF-DMS 1720402
