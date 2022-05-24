
def check_update(data,history):
    new_data = None

    # print(data)
    # print(history)

    if(len(history)!= 0):
        last_data = history[-1]
        res = last_data['response']
        if(res == 'Please enter the source'):
            text = data['Entities']['source']
            last_data['Entities']['source'] = text
            new_data = last_data
        
        elif(res == 'Please enter the destination'):
            text = data['Entities']['dest']
            last_data['Entities']['dest'] = text
            new_data =  last_data

        else:
            new_data = data
    else:
        new_data = data

    return new_data