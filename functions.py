
def get_phone_dicts(strPhoneIDFile, num_phones = 39):

    phone_to_ID, ID_to_phone = {}, {}
    if num_phones == 61:
        ID_col = 3
    elif num_phones == 48:
        ID_col = 4
    elif num_phones == 39 or num_phones == 40:
        ID_col = 5
    else:
        print("Valid values for num_phones are 39, 48 and 61.")

    # Step through the phone ID file and read in the ID for each phone
    with open(strPhoneIDFile, 'r') as f:
        for line in f:
            phone_to_ID[line.split()[0]] = int(line.split()[ID_col])
            if int(line.split()[ID_col]) not in ID_to_phone:
                ID_to_phone[int(line.split()[ID_col])] = line.split()[0]

    return phone_to_ID, ID_to_phone

