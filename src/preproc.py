import numpy as np

class Preprocessor():
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

    def getMatrix(self):
        return self.matrix
    
    def getAttributes(self):
        return self.attributes
    
    def getCountries(self):
        return self.countries
    
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

    def isFilled(self, row):
        ln = np.shape(row)[0]
        for i in range(ln):
            if row[i] == None:
                return False
        
        return True

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


