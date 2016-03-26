import csv
import sys
import math
import random
import subprocess
import numpy

def  Leitor(File, string):
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
                        point=acerta(contador,point)

                pointsin=Point(point)
                if(string=="train"):
                    defsample(pointsin, row)


                pointreturn.append(pointsin)
            l=l+1
            if(string=="test"):
                device.append(row[0])

    return pointreturn

def defsample(point,row):
    global sample
    if (row[1]=="A001"):
        sample[0]=point
    if (row[1]=="A002"):
        sample[1]=point
    if (row[1]=="A003"):
        sample[2]=point
    if (row[1]=="A004"):
        sample[3]=point
    if (row[1]=="A005"):
        sample[4]=point
    if (row[1]=="A006"):
        sample[5]=point
    if (row[1]=="A007"):
        sample[6]=point
    if (row[1]=="A008"):
        sample[7]=point
    if (row[1]=="A009"):
        sample[8]=point
    if (row[1]=="A010"):
        sample[9]=point
    if (row[1]=="A011"):
        sample[10]=point
    if (row[1]=="A012"):
        sample[11]=point
    if (row[1]=="A013"):
        sample[12]=point
    if (row[1]=="A014"):
        sample[13]=point
    if (row[1]=="A015"):
        sample[14]=point
    if (row[1]=="A016"):
        sample[15]=point
    if (row[1]=="A017"):
        sample[16]=point
    if (row[1]=="A018"):
        sample[17]=point
    if (row[1]=="A019"):
        sample[18]=point
    if (row[1]=="A020"):
        sample[19]=point
    if (row[1]=="A021"):
        sample[20]=point
    if (row[1]=="A022"):
        sample[21]=point
    if (row[1]=="A023"):
        sample[22]=point
    if (row[1]=="A024"):
        sample[23]=point
    if (row[1]=="A025"):
        sample[24]=point
    if (row[1]=="A026"):
        sample[25]=point
    if (row[1]=="A027"):
        sample[26]=point
    if (row[1]=="A028"):
        sample[27]=point
    if (row[1]=="A029"):
        sample[28]=point
    if (row[1]=="A030"):
        sample[29]=point
    if (row[1]=="A031"):
        sample[30]=point
    if (row[1]=="A032"):
        sample[31]=point
    if (row[1]=="A033Front"):

        sample[32]=point
    if (row[1]=="A033Parking"):

        sample[33]=point

def acerta(posicao,point):
    if (posicao==0):
        if (point[2]!=-1):
            if(point[3]!=-1):
                if(point[6]!=-1):
                    point[0]=(point[2]+point[3]+point[6])/3

                else:
                    acerta(6,point)
            else:
                acerta(3,point)
        else:
            acerta(2,point)
    elif (posicao==1):
        if(point[3]!=-1):

            if(point[7]):
                   point[1]=(point[3]+point[7])/2
            else:
                   acerta(7,point)


        else:
           acerta(3,point)

    elif (posicao==2):
    	if(point[5]!=-1):
    		if(point[6]!=-1):
    			point[2]=(point[5]+point[6])/2
    		else:
    			acerta(6,point)
    	else:
    		acerta(5,point)

    elif (posicao==3):
    	if(point[2]!=-1):
    		if(point[4]!=-1):
    			point[3]=(point[2]+point[4])/2
    		else:
    			acerta(4,point)
    	else:
    		acerta(2,point)

    elif (posicao==4):
    	if(point[7]!=-1):
    		if(point[8]!=-1):
    			point[4]=(point[7]+point[8])/2
    		else:
    			acerta(8,point)
    	else:
    		acerta(7,point)

    elif (posicao==5):
    	if(point[18]!=-1):
    		if(point[11]!=-1):
    			point[5]=(point[18]+point[11])/2
    		else:
    			acerta(11,point)
    	else:
    		acerta(18,point)

    elif (posicao==6):
    	if(point[1]!=-1):
    		if(point[11]!=-1):
    			point[6]=(point[1]+point[11])/2
    		else:
    			acerta(11,point)
    	else:
    		acerta(1,point)

    elif (posicao==7):
    	if(point[3]!=-1):
    		if(point[10]!=-1):
    			point[7]=(point[3]+point[10])/2
    		else:
    			acerta(10,point)
    	else:
    		acerta(3,point)

    elif (posicao==8):
    	if(point[7]!=-1):
    		if(point[14]!=-1):
    			point[8]=(point[7]+point[14])/2
    		else:
    			acerta(14,point)
    	else:
    		acerta(7,point)

    elif (posicao==9):
    	if(point[1]!=-1):
    		if(point[12]!=-1):
    			point[9]=(point[1]+point[12])/2
    		else:
    			acerta(12,point)
    	else:
    		acerta(1,point)

    elif (posicao==10):
    	if(point[1]!=-1):
    		if(point[13]!=-1):
    			point[10]=(point[1]+point[13])/2
    		else:
    			acerta(13,point)
    	else:
    		acerta(1,point)

    elif (posicao==11):
    	if(point[9]!=-1):
    		if(point[17]!=-1):
    			point[11]=(point[9]+point[17])/2
    		else:
    			acerta(9,point)
    	else:
    		acerta(17,point)

    elif (posicao==12):
    	if(point[6]!=-1):
    		if(point[15]!=-1):
    			point[12]=(point[6]+point[15])/2
    		else:
    			acerta(6,point)
    	else:
    		acerta(15,point)

    elif (posicao==13):
    	if(point[12]!=-1):
    		if(point[14]!=-1):
    			point[13]=(point[12]+point[14])/2
    		else:
    			acerta(14,point)
    	else:
    		acerta(12,point)

    elif (posicao==14):
    	if(point[8]!=-1):
    		if(point[18]!=-1):
    			point[14]=(point[8]+point[18])/2
    		else:
    			acerta(18,point)
    	else:
    		acerta(8,point)

    elif (posicao==15):
    	if(point[16]!=-1):
    		if(point[17]!=-1):
    			point[15]=(point[16]+point[17])/2
    		else:
    			acerta(17,point)
    	else:
    		acerta(16,point)

    elif (posicao==16):
    	if(point[13]!=-1):
    		if(point[17]!=-1):
    			point[16]=(point[13]+point[17])/2
    		else:
    			acerta(17,point)
    	else:
    		acerta(13,point)

    elif (posicao==17):
    	if(point[5]!=-1):
    		if(point[18]!=-1):
    			point[17]=(point[5]+point[18])/2
    		else:
    			acerta(18,point)
    	else:
    		acerta(5,point)

    elif (posicao==18):
    	if(point[13]!=-1):
    		if(point[16]!=-1):
    			point[18]=(point[13]+point[16])/2
    		else:
    			acerta(16,point)
    	else:
    		acerta(13,point)
    return point

def main():

    arquivosaida=open("saida.txt","w")

    np = 100
    dimension=19
    lower = 0
    upper = 100
    ncluster=35
    opt_cut = 0.001
    #points = [makeRandomPoint(dimension, lower, upper) for i in xrange(np)]

    '''clusters = kmeans(points, ncluster, opt_cut)
    for i,c in enumerate(clusters):
            for p in c.points:
                print "Cluster: ", i, "\Point :", p
'''
    ID = kmeans(ncluster, opt_cut)
    for i,c in enumerate(ID):
        for p in c:
            i1=i+1
            p1=p+1
            d=str(device[p])
            istring=str (i1)
            pstring=str(p1)
            string1= d + ","+istring+","+pstring+"\n"
            arquivosaida.write(string1)
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

def kmeans(k, cutoff):

    # Pick out k random points to use as our initial centroids
    #initial = random.sample(points, k)
    pointsT= Leitor("train.csv","train")
    pointsR= Leitor("test.csv","test")
    global sample
    initial = sample

    # Create k clusters using those centroids
    clusters = [Cluster([p]) for p in initial]

    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster

        lists = [ [] for c in clusters]
        ID=[[] for c in clusters]
        clusterCount = len(clusters)

        # Start counting loops
        loopCounter += 1
        # For every point in the dataset ...

        for p in pointsT:

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


        t=0;
        for p in pointsR:
            t=t+1
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
            ID[clusterIndex].append(t)
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
    return ID

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
device=[]
zero=Point([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
sample=[zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero]
main()
