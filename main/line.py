import numpy as np
class Line():

    def __init__ (self):

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

    def __calcBestFit(self):
        self.bestx = np.mean(np.array(recent_xfitted),axis = 0)

    def addLine(self,fitX , y , coeff , rCurv):
        self.allx = fitX
        self.ally = y
        self.detected = True
        self.current_fit = np.copy(coeff)
        self.radius_of_curvature = rCurv
        self.recent_xfitted.append(fitX)
        self.__calcBestFit()
        return self.best_fit
