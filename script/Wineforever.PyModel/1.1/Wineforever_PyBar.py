class Bar:
    def __init__(self,name,max):
        self.name = name
        self.max = max
        self.i = 1
    def next(self):
        percent = self.i/self.max
        progress = '\r' + self.name + ': ['
        for i in range(0,50):
            if percent >= 0.02 * i:
                progress += "#"
            else:
                progress += "-"
        progress += "](" + str(round(percent*100,2)) + "%)"
        self.i = self.i + 1
        print(progress,end="")
        if percent==1:
            self.i = 1
            print()