# Disaster Response Pipeline Project

### Project Promotion  

Figure Eight, formerly CrowdFlower, now Appen, was a human-in-the-loop machine learning and artificial intelligence company based in San Francisco. The project provided by this compaby, requires to apply the data engineering skills to expand the opportunities and potential as a data scientist skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.    
This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.   
This project will show off the software skills, including the ability to create basic data pipelines and write clean, organized code!

### The file structure
```
- app  
| - template  
|   |- master.html  # main page of web app  
|   |- go.html  # classification result page of web app  
|- run.py  # Flask file that runs app  
|- Classify_message.png  
|- Trainingdata_overview_screenshot.png  

- data  
|- disaster_categories.csv  # data to process   
|- disaster_messages.csv  # data to process  
|- process_data.py  
|- DisasterResponse.db   # database to save clean data to  

- models  
|- train_classifier.py  
|- classifier.pkl  # saved model   

- README.md  
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### The view from web app 

[Trainingdata_overview](https://github.com/XueWang2019/Disaster_Response_Pipeline/blob/master/app/Trainingdata_overview_screenshot.png)

[Classified message](https://github.com/XueWang2019/Disaster_Response_Pipeline/blob/master/app/Classify_message.png)

