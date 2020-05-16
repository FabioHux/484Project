import numpy as np

class Preprocessor():

    """
    Function that sorts the file input into a formatted matrix

    Input: File pointer, not a string of the file that has been read
    ***FILE WILL BE READ INSIDE PREPROCESS***

    Formatting requirements of the information:
    -Must be a csv file
    -First line must contain attributes
    -First column must be the countries
    -Second column ***WILL*** be skipped
    -Every data block that isn't in the country column will be set to a float or None if it cannot be converted

    Three values will be stored in the instance of preprocess: Matrix with the information, list of attributes from the second column on, list of each row's respective country
    """
    def preprocess(self, file):
        entries = file.readlines()

        self.matrix = np.empty((0,len(entries[0].split(",")) - 2))
        self.countries = np.empty(0)
        self.attributes = np.asarray(entries[0].split(",")[2:])

        for i in range(1, len(entries)):
            line_el = entries[i].split(",")
            arr = np.empty(0)
            for j in range(len(line_el)):
                if j == 0:
                    self.countries = np.append(self.countries, line_el[j])
                elif j >= 2:
                    try:
                        arr = np.append(arr, float(line_el[j]))
                    except:
                        arr = np.append(arr, None)
                    
            self.matrix = np.append(self.matrix, [arr], axis = 0)

    """
    Getter function for preprocess's matrix
    """
    def getMatrix(self):
        return self.matrix
    
    """
    Getter function for preprocess's attributes
    """
    def getAttributes(self):
        return self.attributes
    
    """
    Getter function for preprocess's countries
    NOTE THAT THIS LIST CONTAINS DUPLICATES AS IT REPRESENTS THE MATRIX'S RESPECTIVE COUNTRY FOR EACH ROW
    """
    def getCountries(self):
        return self.countries
    
    """
    Function to remove a column in the matrix based on the given attribute

    Function will take in an attribute and return the column given for that attribute while removing the respective data from the matrix and attributes list
    """
    def getColumn(self, attribute = None):
        if attribute == None:
            return None
        col = 0
        for i in range(np.shape(self.attributes)[0]):
            if self.attributes[i] == attribute:
                col = i
                break
        ret = self.matrix[...,col]
        self.matrix = np.hstack((self.matrix[...,0:col], self.matrix[...,col + 1:]))
        self.attributes = np.hstack((self.attributes[...,0:col], self.attributes[...,col + 1:]))
        return ret
    
    """
    Function to insert a column in the list

    Must be given an attribute as to be able to remove it later on.

    Column must be the same height as the matrix's column
    """
    def insertColumn(self, col, attribute):
        self.attributes = np.append(self.attributes, attribute)
        self.matrix = np.vstack((self.matrix, [col]))

    """
    Function that returns a boolean stating whether the entry given has all of its columns filled out.
    """
    def isFilled(self, row):
        ln = np.shape(row)[0]
        for i in range(ln):
            if row[i] == None:
                return False
        
        return True

    """
    Function that modifies the Preprocess's matrix to remove all rows that have entries with unfilled data using the isFilled function (see isFilled).

    Will also remove the respective entry from the list of countries
    """
    def cleanUnfilled(self):
        nMatrix = np.empty((0,np.shape(self.matrix)[1]))
        nCountries = np.empty(0)

        for i in range(np.shape(self.matrix)[0]):
            if self.isFilled(self.matrix[i]):
                nMatrix = np.append(nMatrix, [self.matrix[i]], axis = 0)
                nCountries = np.append(nCountries, self.countries[i])
        
        self.matrix = nMatrix
        self.countries = nCountries
    
    #def removeCountry(self, country_name):


