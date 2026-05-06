## Setup

```
conda create -n ie python=3.10
pip install -r requirements.txt
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install soundfile # if you don't have an audio backend already
```

If you have trouble with setting up the environment, please get in touch with the course staff. Note: you can reuse your environment from Project 2.

## Usage

### Training

```
cd IE_Proj3_2026
python train.py
```

The outline of train.py is as follows:

* Extracts normalized MFCC features for each utterance;
* The model takes normalized MFCC features and produces a prediction lattice;
* A CTC loss will be computed for the ground truth transcript marginalized over all possible paths in the prediction lattice. You will implement CTC loss yourself, how exciting!

You do not have to edit train.py in any way (apart fron hyper-parameters like learning rate). Do not change the interface to train.py: we will attempt to run train.py without any parameters on Gradescope.

Note: similar to Project 2, train.py enforces a 20-minute time limit on training. **Please do not change the time limit** as doing so could make your submission timeout on Gradescope. 

### Inference

```
python infer.py
```

infer.py reads best_model.pt and writes the predictions from data/clsp.devwav: output.txt, output.txt.greedy (greedy decoding), and output.txt.beam (beam search). You can change the checkpointing logic by editing the training loop in train.py.

## To-do Items

### Improve the Model

This is not the focus of Project 3 so you don't need to worry about it too much. If your model for Project 2 had good accuracy, then feel free to reuse it for Project 3. If you're not satisfied with your model from Project 2, you can reach out to the course staff for a reference model.

### Implement CTC Loss

CTC has been covered in class, so we won't repeat the theory here. Please implement CTC loss from scratch, without calling existing implementations such as torch.nn.functional.ctc_loss. Hint: the decoding method from Project 2 bears many similarities to CTC loss computation.

Note: if you run into any technical challenges, such as numerical instability issues, feel free to reach out to the course staff or post your question to Canvas if you think it might be instructive. We'll do out best to help you.

### Implement Decoding

Similar to Project 2, implement methods that convert your model's prediction lattices into word predictions.

- CTC loss-based decoding, which computes the CTC loss for each of the words in our word list and finds the one with the lowest CTC loss. For the sake of training efficiency, we won't report this number during training, but you're welcome to add it back yourself if you're confident in the speed of your CTC loss implementation. 
- Greedy search, which simply concatenates the model's prediction on each frame, then collapses repeats and blanks according to CTC rules.
- Beam search (optional), which at each time interval, keeps track of k running hypotheses with the highest forward probabilities. When implementing beam search, it is worth keeping in mind the following questions:
    
    - What does a beam hypothesis consist of?
    - At the end of beam search, the beam with the highest forward probability is chosen. Why is it different from the result of greedy search?

Once you implement the above, your training script should start producing valid character error rates (CER), which you can use as a reference for the quality of your model.

## Submission

Please make sure the following files are contained within your submission:

```
train.py
infer.py
best_model.pt
output.txt
output.txt.greedy
output.txt.beam (if you implemented beam search)
utils/features.py
utils/decode.py
modules/dataset.py
modules/model.py
```
As well as any additional dependencies that you added. You do not have to include your data files in your submission.

## Evaluation Criteria

You will be graded on the following:

* Presence of required files.
* Your character error rate as computed from output.txt.greedy and output.txt.beam (if you chose to implement beam search).
* Your character error rate as generated from best_model.pt and infer.py.
* Your character error rate from best_model.pt trained using your train.py within a time limit of 20 minutes.

The test set is hidden from you. Please do not try to find it online.

**Please avoid using any pre-trained models in your submission for part 1 of Project 3.**
