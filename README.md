All the files are for my report "Use of the adjoint method for the efficient calculation of sensitivities" in MCF of Oxford
The main contributions are applying exploring the use of JAX in automatic differentiation and developing the reverse mode analysis of the bicubic spline including its creation, evaluation and application.

My personal work is mainly in the three Jupyter notebook files "basic_reverse_mode.ipynb", "checkpointing.ipynb" and "bicubic.ipynb".
One of the main contributions in the dissertation is to develop the forward and reverse mode of a 2D spline and its applications, which are based on the rest Python files. Those Python files (jax_spline.py, jax_spline_test.py, jax_trid.py, jax_trid_test.py, spline.py, spline_test.py, trid.py, trid_test.py) are implemented by Professor Mike Giles and based on the book [1] written by himself.

Thanks for the kindest support from my supervisor Mike.

[1] M.B. Giles. Smoking adjoints – the book (draft), 2024. personal communication
