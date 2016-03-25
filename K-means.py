import csv
import sys
import math
import random
import subprocess
import numpy

def  Leitor(File):
    maxi=0;
    with open(File, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        l=0

        pointreturn=[]

        for row in reader :
            if(l!=0 ):
                i=2
                point=[]
                pointsin=[]
                while (i<21):
                    if(row[i]==""):
                        row[i]=-1;
                    p=float(row[i])
                    if ((p<=0) ):

                        point.append(p)
                    i=i+1
                contador=-1
                for certo in point:
                    contador=contador+1
                    if (certo ==-1):
                        certo=acerta(contador,point)

                pointsin=Point(point )

                    #print pointsin

                pointreturn.append(pointsin)
            l=l+1

    return pointreturn

def acerta(posicao,point):
    if (posicao==0):
        if (point[2]!=-1):
            if(point[3]!=-1):
                if(point[6]!=-1):
                    point[0]=(point[2]+point[3]+point[6])/3

                else:
                    acerta(6,points)
            else:
                acerta(3,points)
        else:
            acerta(2,points)
    elif (posicao==1):

    '''elif (posicao==2):

    elif (posicao==3):

    elif (posicao==4):

    elif (posicao==5):

    elif (posicao==6):

    elif (posicao==7):

    elif (posicao==8):

    elif (posicao==9):

    elif (posicao==10):

    elif (posicao==11):

    elif (posicao==12):

    elif (posicao==13):

    elif (posicao==14):

    elif (posicao==15):

    elif (posicao==16):

    elif (posicao==17):

    elif (posicao==18):'''
    return

def main():
    points= Leitor("train.csv")

    np = 100
    dimension=19
    lower = 0
    upper = 100
    ncluster=33
    opt_cut = 15
    #points = [makeRandomPoint(dimension, lower, upper) for i in xrange(np)]

    clusters = kmeans(points, ncluster, opt_cut)
    #for i,c in enumerate(clusters):
            #for p in c.points:
                #print "Cluster: ", i, "\Point :", p

class Point:
    '''
    An point in n dimensional space
    '''
    def __init__(self, coords):
        '''
        coords - A list of values, one per dimension
        '''

        self.coords = coords
        self.n = len(coords)

    def __repr__(self):
        return str(self.coords)

class Cluster:
    '''
    A set of points and their centroid
    '''

    def __init__(self, points):
        '''
        points - A list of point objects
        '''

        if len(points) == 0: raise Exception("ILLEGAL: empty cluster")
        # The points that belong to this cluster
        self.points = points

        # The dimensionality of the points in this cluster
        self.n = points[0].n

        # Assert that all points are of the same dimensionality
        for p in points:
            if p.n != self.n: raise Exception("ILLEGAL: wrong dimensions")

        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)

    def update(self, points):
        '''
        Returns the distance between the previous centroid and the new after
        recalculating and storing the new centroid.
        '''
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)

        # Get a list of all coordinates in this cluster
        coords = [p.coords for p in self.points]


        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]

        return Point(centroid_coords)

def kmeans(points, k, cutoff):

    # Pick out k random points to use as our initial centroids
    initial = random.sample(points, k)

    # Create k clusters using those centroids
    clusters = [Cluster([p]) for p in initial]

    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [ [] for c in clusters]
        clusterCount = len(clusters)

        # Start counting loops
        loopCounter += 1
        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = getDistance(p, clusters[0].centroid)

            # Set the cluster this point belongs to
            clusterIndex = 0

            # For the remainder of the clusters ...
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = getDistance(p, clusters[i+1].centroid)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is, and set the point to belong
                # to that cluster
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            lists[clusterIndex].append(p)

        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0

        # As many times as there are clusters ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)

        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print "Converged after %s iterations" % loopCounter
            break
    return clusters

def getDistance(a, b):
    '''
    Euclidean distance between two n-dimensional points.
    Note: This can be very slow and does not scale well
    '''
    if a.n != b.n:

        raise Exception("ILLEGAL: non comparable points")

    ret = reduce(lambda x,y: x + pow((a.coords[y]-b.coords[y]), 2),range(a.n),0.0)
    return math.sqrt(ret)

def makeRandomPoint(n, lower, upper):
    '''
    Returns a Point object with n dimensions and values between lower and
    upper in each of those dimensions
    '''
    p = Point([random.uniform(lower, upper) for i in range(n)])
    return p


main()
