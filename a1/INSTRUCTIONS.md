# Instructions for Assignment 1

## Code

You need to write code to implement matrix multiplication.  A set of tests has already been specified, and you need to ensure that your implementation passes all the tests.  

## Documentation

Your code should be properly documented.  This means that you add comments in the code to indicate the basic approach you have taken to solve the problem.  Only things that are not immediately obvious should be in the comments, do not add comments like "increment counter", "run for loop" etc.  

Instead, you should be able to indicate the functionality of each `for` loop, perhaps with a comment like "loop over all diagonal elements in the first matrix".  

In addition, you should also create a file with the name `README.md` that specifies your algorithm.  For this assignment the README will be quite simple: a description of the approach you have taken for matrix multiplication, and what are the limitations of the solution.  For example, how do you handle invalid inputs, do you check for any special cases or corner cases, what impact do these checks have on the overall performance, etc.

## Submission

You need to submit a single file on Moodle, and this has to be in `.zip` format.  That is, it is a compressed archive of all the files you have.  Please note that the format for submitting this is important - the TAs will be running automatic scripts to download and check your code: if it is not in the correct format your submission will not get evaluated properly.  

For the first assignment, you may be given additional chances to submit, but for future assignments, if your submission cannot be automatically evaluated by the TAs using scripts, you will lose marks.  So it is essential that you do this properly.

### Procedure to generate submission file

The Instructions below assume you are working on the Jupyter server jup.dev.iitm.ac.in.  You can also do this on your own system, you just need to make sure the format of the files and folder are correct.

- Step 1: Create a folder with your roll number in lower case.  For example, if your roll number is EE42B123, use the command `mkdir ee42b123`.  You can do this anywhere - need not be inside the folder where you have the files.

- Step 2: Copy your files into the submission folder.  The following files should be copied (commands are given here for convenience, but in general use whatever means you want to copy files into the folder):
  
  - `cp README.md ee42b123/`
  - `cp test_matmul.py ee42b123/`
  - `cp matmul.py ee42b123/`

  NOTE: you need NOT copy this `INSTRUCTIONS.md` file.  Also avoid putting any other unnecessary files, including `__pycache__`, `*.pyc` (compiled Python code) etc.

- Step 3: Create the zip file: `zip -r ee42b123.zip ee42b123/`

- Step 4: Upload the file `ee42b123.zip` on Moodle for Assignment 1.


