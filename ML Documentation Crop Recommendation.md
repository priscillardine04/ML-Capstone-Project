# Capstone-Project
## Crop Recommendation Documentation

### Project Description
Crop recommendation feature built in for Capstone Project. The system built on Machine Learning techniques to recommend suitable crop based on some parameter conditions that inputted by user. The model deployed in a Pickle format for next deployment to Android Studio. This project was created by Priscilla Ardine Puspitasari, Muhammad Hafizh Rachman, and Qanita Zafa Ariska as part of the Bangkit Capstone Project, demonstrating their skills and knowledge gained throughout the program.
### Features
Giving crop recommendation based on several variables related to soil contents and weather parameters.
### Splitting Data
There are a total of 24.201 row of data. From this dataset, we split the data into 2 set, that are Training set with 80% proportion and Test set with a proportion of 20%
### Inputing Data
Data in the form of table (csv format) that contain ratio of content in soil and weather parameters with 22 label based on plant species
### Build Model
The model that we choose is Random Forest. Random Forest itself is a supervised machine learning algorithm that is popular for Classification and Regression case.

``` python
RanFor = RandomForestClassifier(n_estimators=10, random_state=0)
RanFor.fit(x_train,y_train)
PVRF = RanFor.predict(x_test)
met_RanFor = metrics.accuracy_score(y_test, PVRF)
empt1.append(met_RanFor)
empt2.append(RanFor)
print(met_RanFor)
print(classification_report(y_test,PVRF))
```

![alt text](https://github.com/priscillardine04/ML-Capstone-Project/blob/main/Output%20Model/classification%20metrics.png?raw=true)

Random Forest give an accuracy about 0.99 or 99% for the test set. So, the Random Forest algorithm can give crop recommendation accurately according to the content of soil and weather parameters.

### Prediction
We test the model by trying to predict the crop that suitable with the input variable.
``` python
input_kandungan = np.array([[0, 0, 31, 27, 93, 0, 150]])
rekomendasi = RanFor.predict(input_kandungan)
print(rekomendasi)
```
From that input, the model start doing prediction and giving the following output :
![alt text](https://github.com/priscillardine04/ML-Capstone-Project/blob/main/Output%20Model/crop%20prediction.png?raw=true)
The model giving a recommendation of coconut crop.

### Convert
After choosing this Random Forest algorithm, the model will be converted to Pickle format. Then, the model will be updated by API and deployment will carried out on Android.

### Contact
For any inquiries or further information, please contact the project developers:

- Priscilla Ardine Puspitasari: [Email](mailto:priscillaardine9784@gmail.com)
- Muhammad Hafizh Rachman: [Email](m.hafizh272@gmail.com)
- Qanita Zafa Ariska: [Email](qanitazafa@gmail.com)

Model in Pickle : [Pickle File](https://drive.google.com/file/d/1-vRWTl83uo7ckbBjKQMPPANfZPjLVlSl/view?usp=sharing)
