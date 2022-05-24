import random
import logging

logger = logging.getLogger(__name__)


# data = {"text": "Hi",
#     "Intents": "atis_flight",
#         "Entities": {"source":"buffalo", "dest":"oakland"}}

greetings= ["hi", "hello", "hey", "helloo", "hellooo", "g morining", "gmorning", "good morning",
            "morning", "good day", "good afternoon", "good evening", "greetings",
            "greeting", "good to see you", "its good seeing you", "how are you",
            "how're you", "how are you doing", "how ya doin'", "how ya doin",
             "how is you", "how's you", "how is it going", "how's it going", "how's it goin'",
            "how's it goin", "what is up",  "g’day", "howdy"]

farewell=["Thank you","Bye", "Good day","good bye","thanks","have a good day","cheers","thank you so much for your help","see you",
         "bye-bye", "have a good day"]


def get_dialogue(data):

    logger.info("Inside Dialogue Flow")
    
    intents = data['intent']
    entities = data['Entities']
    output= None

    input_query = data['userinput']

    if input_query.lower() in greetings:
        intents='greetings'
        entities={}
    elif input_query.lower() in farewell:
        intents='farewell'
        entities={}
    else:
        logger.info('Checking other intents by calling get_intents and fetching the NERs from get_ner method')
        

    logger.info("Intent is: "+intents)
    source = None
    dest = None
    time_dt = None

    flights = ['Panam Airlines', 'American Airlines', 'Virgin Airlines', 'United Airlines', 
                'Breeze Airlines','Alaska Airlines','Frontier Airlines', 'JetBlue','Air India','British Airways','ABC Airlines']

    aircraft_types = ['Airbus A320 family', 'Boeing 737 NG', 'Boeing 777','Airbus A330','Boeing 747','Airbus A319','Airbus A3303','Airbus 550M','Boeing 007']

    flight_time = list(range(1,13))

    flight_fares = list(range(100,1000,50))


    if len(entities)!= 0:
        if 'source' in entities.keys():
            source = entities['source']
        
        if 'dest' in entities.keys():
            dest = entities['dest']
        
        if 'TIME' in entities.keys():
            time_dt = entities['TIME']
        
    
    if intents == 'greetings':
        output='Hello! I am Air Travel Information Bot. I can help you with flight details, airfares, type of flights.'

    elif intents== 'farewell':
        output='Thank you for using ATIS chat bot! Have a good day. Please click the Quit button to exit.'

    elif intents == 'atis_flight':
        if source!= None and dest!= None and source!='' and dest!='' and time_dt!= None:
            random_flights = random.sample(flights, 3)
            #random_time = random.sample(flight_time, 3)
            output = "Below are the flights from "+source+" to "+dest+" for "+ time_dt + " time :"+random_flights[0]+", "+random_flights[1]+", "+random_flights[2]+"."
        
        elif source!= None and dest!= None and source!='' and dest!='':
            random_flights = random.sample(flights, 3)
            random_time = random.sample(flight_time, 3)
            output = "Below are the flights from "+source+" to "+dest+" :"+random_flights[0]+" at "+str(random_time[0])+"PM, "+random_flights[1]+" at "+str(random_time[1])+"PM, "+random_flights[2]+" at "+str(random_time[2])+"PM."
        
        elif source == None or source == '':
            if dest!=None and dest!='':
                output = "Please enter the source"
            elif dest == None or dest == '':
                output = "Sorry!! Please enter the source and destination in your details"
        
        elif dest == None or dest == '':
            if source!=None and source!='':
                output = "Please enter the destination"
            elif dest == None or dest == '':
                output = "Sorry!! Please enter the source and destination in your details"
      

    elif intents == 'atis_flight_time':
        if source!= None and dest!= None and source!='' and dest!='' and time_dt!= None:
            random_flights = random.sample(flights, 3)
            #random_time = random.sample(flight_time, 3)
            output = "Below are the flights from "+source+" to "+dest+" for "+ time_dt + " time :"+random_flights[0]+", "+random_flights[1]+", "+random_flights[2]+"."
        
        elif source!= None and dest!= None and source!='' and dest!='':
            x = "Here are the timings and flights available for today "
            random_flights = random.sample(flights, 3)
            random_time = random.sample(flight_time, 3)
            random_time.sort()
            output = x+"from "+source+" to "+dest+": "+random_flights[0]+" at "+str(random_time[0])+"PM, "+random_flights[1]+" at "+str(random_time[1])+"PM, "+random_flights[2]+" at "+str(random_time[2])+"AM."
        
        elif source == None or source == '':
            if dest!=None and dest!='':
                output = "Please enter the source"
            elif dest == None or dest == '':
                output = "Sorry!! Please enter the source and destination in your details"
        
        elif dest == None or dest == '':
            if source!=None and source!='':
                output = "Please enter the destination"
            elif dest == None or dest == '':
                output = "Sorry!! Please enter the source and destination in your details"
        
    
    elif intents == 'atis_airline':
        if source!= None and dest!= None and source!='' and dest!='' and time_dt!= None:
            random_flights = random.sample(flights, 3)
            #random_time = random.sample(flight_time, 3)
            output = "Below are the flights from "+source+" to "+dest+" for "+ time_dt + " time :"+random_flights[0]+", "+random_flights[1]+", "+random_flights[2]+"."
        
        elif source!= None and dest!= None and source!='' and dest!='':
            x = "Here are the flights available"
            random_flights = random.sample(flights, 4)
            output = x+" from "+source+" to "+dest+": "+random_flights[0]+", "+random_flights[1]+", "+random_flights[2]+", "+random_flights[3]+"."
        
        elif source == None or source == '':
            if dest!=None and dest!='':
                output = "Please enter the source"
            elif dest == None or dest == '':
                output = "Sorry!! Please enter the source and destination in your details"
        
        elif dest == None or dest == '':
            if source!=None and source!='':
                output = "Please enter the destination"
            elif dest == None or dest == '':
                output = "Sorry!! Please enter the source and destination in your details"
        
    
    elif intents == 'atis_airfare':
        if source!= None and dest!= None and source!='' and dest!='' and time_dt!= None:
            x = "These are the flights and price I could find"
            random_flights = random.sample(flights, 4)
            random_fares = random.sample(flight_fares, 4)
            #random_time = random.sample(flight_time, 3)
            output = x+"from "+source+" to "+dest+" for "+ time_dt + " time :"+random_flights[0]+": "+str(random_fares[0])+" USD, "+random_flights[1]+": "+str(random_fares[1])+" USD, "+random_flights[2]+": "+str(random_fares[2])+" USD, "+random_flights[3]+": "+str(random_fares[3])+" USD."
        
        elif source!= None and dest!= None and source!='' and dest!='':
            x = "These are the flights and price I could find"
            random_flights = random.sample(flights, 4)
            random_fares = random.sample(flight_fares, 4)
            output = x+"from "+source+" to "+dest+": "+random_flights[0]+": "+str(random_fares[0])+" USD, "+random_flights[1]+": "+str(random_fares[1])+" USD, "+random_flights[2]+": "+str(random_fares[2])+" USD, "+random_flights[3]+": "+str(random_fares[3])+" USD."
        
        elif source == None or source == '':
            if dest!=None and dest!='':
                output = "Please enter the source"
            elif dest == None or dest == '':
                output = "Sorry!! Please enter the source and destination in your details"
        
        elif dest == None or dest == '':
            if source!=None and source!='':
                output = "Please enter the destination"
            elif dest == None or dest == '':
                output = "Sorry!! Please enter the source and destination in your details"
        
        
    elif intents == 'atis_aircraft':
        if source!= None and dest!= None and source!='' and dest!='' and time_dt!= None:
            x = "Here are the list of types of flights used "
            random_flights = random.sample(aircraft_types, 3)
            #random_time = random.sample(flight_time, 3)
            output =  x+"from "+source+" to "+dest+" for "+ time_dt + " time :"+random_flights[0]+", "+random_flights[1]+", "+random_flights[2]+"."
        
        elif source!= None and dest!= None and source!='' and dest!='':
            x = "Here are the list of types of flights used "
            random_types = random.sample(aircraft_types, 4)
            output = x+"from "+source+" to "+dest+": "+random_types[0]+": "+random_types[0]+", "+random_types[1]+", "+random_types[2]+", "+random_types[3]+"."
        
        elif source == None or source == '':
            if dest!=None and dest!='':
                output = "Please enter the source"
            elif dest == None or dest == '':
                output = "Sorry!! Please enter the source and destination in your details"
        
        elif dest == None or dest == '':
            if source!=None and source!='':
                output = "Please enter the destination"
            elif dest == None or dest == '':
                output = "Sorry!! Please enter the source and destination in your details"
        
    

    else:
        output = "Sorry, I didn't understand. I can only help you with Flight Details, Flight Fares, Flight timings and types of Flights. Please enter the correct information."
        


    logger.info("End of Dialogue Flow")
        

    print(output)
    return output
    

# get_dialogue();