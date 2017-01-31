import numpy as np
class Line():

    def __init__ (self,buffer_size):

        self.__buffer_size = buffer_size
        self.__n = 0
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        self.Full = False

    def __calcBestFit(self):
        self.bestx = np.mean(np.array(self.recent_xfitted),axis = 0)

    def addLine(self,fitX , y , coeff , rCurv):
        self.allx = fitX
        self.ally = y
        self.detected = True
        self.current_fit = np.copy(coeff)
        self.radius_of_curvature = rCurv
        if(self.__n < self.__buffer_size and not self.Full):
            self.recent_xfitted.append(fitX)
            self.__n = self.__n + 1
            if(self.__n == self.__buffer_size):
                self.Full = True
                self.__n = 0
        else:
            self.recent_xfitted[self.__n] = fitX
            self.__n = (self.__n +1) % self.__buffer_size

        self.__calcBestFit()
        return self.bestx
