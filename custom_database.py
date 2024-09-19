

def InsertPicture(cursor, picture):
    cursor.execute("INSERT INTO pictures (picture) VALUES (?)", (picture,))
    cursor.commit()
    