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
                    if j == 2:
                        if line_el[j] == "Developing":
                            arr = np.append(arr, 0)
                        elif line_el[j] == "Developed":
                            arr = np.append(arr, 1)
                        else:
                            line_el[j] == None
                    else:
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

    
    #def removeCountry(self, country_name):


