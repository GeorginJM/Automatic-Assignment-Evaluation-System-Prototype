The contents provided above are the files that I have used to create the Automatic Evaluation System. Considering this is an extreamely vast topic , I have taken the previlage to scale it down. Hence, the project i have created perform automatic evaluation for only class 10 Biology CBSE syllabus. The questions i used to train are given in the word document. The question has to be provided as such and the model is capable of detecting whether the received answer is corrected or not. Upon testing, it worked as expected for the most part except that it occasionally faces trouble handling negated sentences of corrected answers. If you wish to perform automatic evaluation for a different subject, i would suggest that you create a different word file for training with the questions and its corresponding answers provided in the same format as the above qanda.docx file. Furthermore , this project uses BERT which is a pretrained LLM for better contextual understanding of the sentences. Transfer Learning is performed as BERT model is fine-tuned on the subject's questions and answers. Tokenisation and NLP procedures are carried out to form a desirable ouput. Threhold is provided as 0.75 as consistent testing proved that the answers with value above 0.75 were more likely correct. 
On the student's side, the students have to provide the questions they were assigned with their corresponding answer in a format similar to sample.docx. The pre-processing extracts the questions and answers for evaluatin by the model. 
The size of the model was too big to be added here. The code for the model is provided in ml.py . Upon running you might come across shape errors in ur devide which you will need to fix. Do note that a device having 16GB RAM is required for running the ml code.
