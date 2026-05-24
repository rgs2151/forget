1. Organize path of stores
2. Load the model to n gpus
3. Take the train and test df, divide it by n gpus and then feed it to the model to create the baseline responses
4. Generate the contrastive refusal response to the baseline and then feed it to the model and catch the activation
5. Calculate the vectors with whatever method mentioned and store them
5.5 using very small randomly sampled tran set, do a sweep of the hyperparams like s
6. organize the steering classes with the calculated vectors, and then shove them to the gpus
7. now simply run the test df through this prepared model and save the results
8. add the score table based on this steered results
9. use the scored csv for the plots and stuffs