# Evaluation of In-Person Counseling Strategies To Develop Physical Activity Chatbot for Women
Dataset and Codebase for Evaluation of In-Person Counseling Strategies To Develop Physical Activity Chatbot for Women, published as a long paper in SIGDIAL 2021.

## Dataset
The simulated dialog is under directory `data/`.  
Since releasing the original interview data is not approved by our IRB and HIPPA, we created 44 dialogs (772 sentences) based on the original interview daialog for our community to use.

## Annotation Scheme
The annotation scheme guidelines can be found under `Annotation Scheme.pdf`.

## Strategy Classifier
The code of the strategy classifier is under directory `classifier/`.  
To use the fine-tuned model for strategy prediction, please download `training_args.bin`from [here](https://drive.google.com/file/d/13GWO8Nwby7MmL15Hq1aUccJ-kQhsYI85/view?usp=sharing) and `optimizer.pt` from [here](https://drive.google.com/file/d/1RCAHRLoq4iujWBRmzHhTwbgpo0w_DEfy/view?usp=sharing) and put them under the `classifier/models/strategy`folder.


