#!/usr/bin/env python
# coding: utf-8


### imports
import os
import math
from itertools import permutations
from multiprocessing import Pool

from tqdm import tqdm, trange

import ephem
from skyfield.api import S, EarthSatellite
from skyfield.api import load
from sgp4.api import Satrec, WGS84

import networkx as nx
from networkx.classes.function import path_weight
import pandas as pd

### constellation utilities

KEP_CONS = 3.9861e14
RUNS = 1 # how many orbiting periods to simulate
STEP = 60 # simulation interval (interval between snapshots) in seconds

ts = load.timescale()

def getMeanMotion(orbitHeight):
    global KEP_CONS
    return ((KEP_CONS**(1./3))/(orbitHeight+ephem.earth_radius))**(3./2)*86400/(2*ephem.pi)

def getSatTrack(satnum, meanAnomaly, raan, epoch, ecc, perig, incl, mm, period):
    satrec = Satrec()
    satrec.sgp4init(
        WGS84,           # gravity model
        'i',             # 'a' = old AFSPC mode, 'i' = improved mode
        satnum,          # satnum: Satellite number
        epoch,           # epoch: days since 1949 December 31 00:00 UT
        2.8098e-05,      # bstar: drag coefficient (/earth radii)
        6.969196665e-13, # ndot: ballistic coefficient (revs/day)
        0.0,             # nddot: second derivative of mean motion (revs/day^3)
        ecc,             # ecco: eccentricity
        perig,           # argpo: argument of perigee (radians)
        incl/180.*ephem.pi, # inclo: inclination (radians)
        meanAnomaly,        # mo: mean anomaly (radians)
        2*ephem.pi*mm/24./60, # no_kozai: mean motion (radians/minute)
        raan, # nodeo: right ascension of ascending node (radians)
    )

    sat = EarthSatellite.from_satrec(satrec, ts)
    return sat.at(period)

def genKey(pair):
    if (not type(pair) is tuple) or (len(pair) != 2) or (pair[0] == pair[1]):
        raise Exception('invalid input')
    if pair[0] > pair[1]:
        return (pair[1], pair[0])
    return pair

def getDistance(posList1, posList2, t, id1, id2, distCache=None): # a and b are arrays of positions, returns kilometer
    key = genKey((id1, id2))
    if distCache != None and key in distCache:
        return distCache[key]
    dist = (posList1[t]-posList2[t]).distance().km
    if distCache != None:
        distCache[key] = dist
    return dist

# the maximum distance between an earth surface point and a satellite for that satellite to be visible, given the orbit height and elevation angle
def getMaxDistance(height, angle):
    r = height+ephem.earth_radius/1000
    h = height*1.0
    arh = (angle+90)/180.0*ephem.pi
    ar = math.asin(r*math.sin(arh)/(r+h))
    return (r+h)*math.sin(ephem.pi-arh-ar)/math.sin(arh)

def getSatId(orbitNum, satNum): # name a satellite
    return 'sat-%d-%d' % (orbitNum, satNum)

def getGtId(city): # name a ground station
    return 'city-%s' % city

def isSameOrNeighborOrbit(sat1, sat2, nOrbits):
    orbitNum1 = sat1.orbitNum
    orbitNum2 = sat2.orbitNum
    return (orbitNum1 in [(orbitNum2-1)%nOrbits, orbitNum2, (orbitNum2+1)%nOrbits])


### constellation

class Satellite:
    def __init__(self, id, orbitN, satN, track):
        self.id = id
        self.orbitNum = orbitN
        self.satNum = satN
        self.track = track

# e.g., starlink = Constellation(oh=550, no=24, ns=66, incl=53, el=25)
class Constellation:
    global RUNS
    global STEP
    global ts

    def __init__(self, oh, incl, no, ns, el, isPi=False, withPhasingDiff=False, withDynamicISLs=False):
        self.ORBIT_HEIGHT = oh # in km
        self.NUM_ORBITS = no # options: 40 (for 40_40_53deg), 34 (kuiper_p1), 72 (starlink_p1)
        self.NUM_SATS_PER_ORBIT = ns # options: 40 (for 40_40_53deg), 34 (kuiper_p1), 22 (starlink_p1)
        self.INCLINATION = incl # options: 53 (for 40_40_53deg), 51.9 (kuiper_p1), 53 (starlink_p1)
        self.MEAN_MOTION = getMeanMotion(self.ORBIT_HEIGHT*1000)
        self.ORBIT_PERIOD = int(1./self.MEAN_MOTION*24*60*60) # in seconds

        self.ELEVATION_ANGLE = el

        self.ECCENTRICITY = 0.001 # circular orbits
        self.ARG_OF_PERIGEE = 0.0 # circular orbits

        T0 = ts.utc(1949, 12, 31, 0, 0, 0)
        self.SIM_PERIOD = int(RUNS * self.ORBIT_PERIOD) # in seconds
        self.PERIOD = ts.utc(2021, 1, 1, 0, 0, range(self.SIM_PERIOD))

        self.MAX_DISTANCE = getMaxDistance(self.ORBIT_HEIGHT, self.ELEVATION_ANGLE)

        # create satellites
        print('Creating satellites...')
        self.satArray = [] # satellite records indexed by orbit and satellite number
        self.satDict = {} # for quick lookup
        for orbitNum in trange(self.NUM_ORBITS):
            self.satArray.append([])
            raanFactor = 2
            if isPi:
                raanFactor = 1
            raan = raanFactor*ephem.pi*orbitNum/self.NUM_ORBITS
            meanAnomalyOffset = 0
            if withPhasingDiff:
                meanAnomalyOffset = (orbitNum%2)/2.0
            for satNum in range(self.NUM_SATS_PER_ORBIT):
                satId = getSatId(orbitNum, satNum)
                meanAnomaly = 2*ephem.pi*(satNum+meanAnomalyOffset)/self.NUM_SATS_PER_ORBIT # simple phasing setting, won't mess with inter-plane ISL that are set according to satellite number
                track = getSatTrack(self.NUM_SATS_PER_ORBIT*orbitNum+satNum, meanAnomaly, raan, self.PERIOD[0]-T0, self.ECCENTRICITY, self.ARG_OF_PERIGEE, self.INCLINATION, self.MEAN_MOTION, self.PERIOD)
                self.satArray[orbitNum].append(Satellite(satId, orbitNum, satNum, track))
                self.satDict[satId] = self.satArray[orbitNum][satNum]
        
        # generate the inter-satellite topology at each epoch (second, according to the interval set with STEP)
        # the weight of an edge (ISL) is set to the distance between two satellites, thus reflects the delay (divide by speed of light)
        print('Setting up ISLs, and generating topology snapshots per %d seconds...'%STEP)
        self.snapshots = {} # each graph represents the snapshot of the topology at an epoch, indexed by the epoch in seconds
        # persistent ISLs following the Grid+ pattern
        distCaches = []
        for t in trange(0, self.SIM_PERIOD, STEP):
            G = nx.Graph()
            distCache = {}
            for orbitNum in range(self.NUM_ORBITS):
                for satNum in range(self.NUM_SATS_PER_ORBIT-1):
                    sat1 = self.satArray[orbitNum][satNum]
                    sat2 = self.satArray[orbitNum][satNum+1]
                    G.add_edge(sat1.id, sat2.id, weight=getDistance(sat1.track, sat2.track, t, sat1.id, sat2.id, distCache))
                    if orbitNum < self.NUM_ORBITS-1:
                        sat2 = self.satArray[orbitNum+1][satNum]
                        G.add_edge(sat1.id, sat2.id, weight=getDistance(sat1.track, sat2.track, t, sat1.id, sat2.id, distCache))
                    else:
                        sat2 = self.satArray[0][satNum]
                        G.add_edge(sat1.id, sat2.id, weight=getDistance(sat1.track, sat2.track, t, sat1.id, sat2.id, distCache))
                G.add_edge(self.satArray[orbitNum][0].id, self.satArray[orbitNum][self.NUM_SATS_PER_ORBIT-1].id,
                           weight=getDistance(self.satArray[orbitNum][0].track, self.satArray[orbitNum][self.NUM_SATS_PER_ORBIT-1].track, t,
                                              self.satArray[orbitNum][0].id, self.satArray[orbitNum][self.NUM_SATS_PER_ORBIT-1].id, distCache))
            self.snapshots[t] = G
            distCaches.append(distCache)
        # dynamic ISLs, a single ISL with a nearby satellite traveling in the opposite direction
        if withDynamicISLs:
            for t in trange(0, self.SIM_PERIOD, STEP):
                G = self.snapshots[t]
                distCache = distCaches[t]
                for satId in self.satDict:
                    distDict = {}
                    sat1 = self.satDict[satId]
                    for tSatId in self.satDict:
                        sat2 = self.satDict[tSatId]
                        if isSameOrNeighborOrbit(sat1, sat2, self.NUM_ORBITS):
                            continue
                        dist = getDistance(sat1.track, sat2.track, t, sat1.id, sat2.id, distCache)
                        distDict[tSatId] = dist
                    distItemList = sorted(distDict.items(), key=lambda x: x[1], reverse=False)
                    tSatId = distItemList[0][0]
                    dist = distItemList[0][1]
                    G.add_edge(satId, tSatId, weight=dist)
                distCaches[t] = distCache
                self.snapshots[t] = G


### an Scenario object binds a constellation to a set of ground stations

class Scenario:
    global STEP

    def __init__(self, constellation, gtDict):
        self.constellation = constellation
        self.gtDict = gtDict

        print('Computing distances from GTs...')
        self.gtTracks = {}
        self.dists = {} # for each gt and epoch, contains only visible satellites and the distance
        for gtId in tqdm(self.gtDict):
            self.gtTracks[gtId] = self.gtDict[gtId].at(self.constellation.PERIOD)
            self.dists[gtId] = {}
            for t in range(0, self.constellation.SIM_PERIOD, STEP):
                satDists = {}
                for satId in self.constellation.satDict:
                    dist = (self.constellation.satDict[satId].track[t]-self.gtTracks[gtId][t]).distance().km
                    if dist < self.constellation.MAX_DISTANCE:
                        satDists[satId] = dist
                self.dists[gtId][t] = satDists

        print('Computing visible orbits...')
        self.visibleOrbits = {}
        empty = False
        for gtId in tqdm(self.gtDict):
            orbits = []
            current = set(range(self.constellation.NUM_ORBITS))
            for t in self.constellation.snapshots:
                now = set()
                for satId in self.getVisibleSats(gtId, t):
                    now.add(self.constellation.satDict[satId].orbitNum)
                tmp = current & now
                if len(tmp) == 0:
                    orbits.append([t, tuple(current)])
                    current = now
                    empty = True
                else:
                    current = tmp
                    empty = False
            if not empty:
                orbits.append([list(self.constellation.snapshots.keys())[-1]+1, tuple(current)])
            self.visibleOrbits[gtId] = orbits

        self.dispatcher = {
            'closest active': self.closestActive,
            'closest lazy': self.closestLazy,
            'orbit closest active': self.orbitClosestActive,
            'orbit closest lazy': self.orbitClosestLazy,
            'orbit closest active T': self.orbitClosestActiveT,
            'orbit closest lazy T': self.orbitClosestLazyT
        }
        self.attachmentsDict = {}
    
    def getVisibleSats(self, gtId, t):
        return list(self.dists[gtId][t].keys())

    def getAttachments(self, strategy):
        if not strategy in self.attachmentsDict:
            print('Determining access satellites using "%s" strategy'%strategy)
            self.attachmentsDict[strategy] = self.dispatcher[strategy]()
        return self.attachmentsDict[strategy]

    def getClosest(self, gtId, t):
        if len(self.dists[gtId][t]) == 0:
            return None
        sortedDists = sorted(self.dists[gtId][t].items(), key=lambda x: x[1], reverse=False)
        return sortedDists[0][0]
    
    def getOrbitClosest(self, curSat, gtId, t):
        if len(self.dists[gtId][t]) == 0:
            return None
        orbitNum = self.constellation.satDict[curSat].orbitNum
        sortedDists = sorted(self.dists[gtId][t].items(), key=lambda x: x[1], reverse=False)
        for item in sortedDists:
            if self.constellation.satDict[item[0]].orbitNum == orbitNum:
                return item[0]
        return None

    def getOrbitClosestT(self, orbitNum, gtId, t):
        if len(self.dists[gtId][t]) == 0:
            return None
        sortedDists = sorted(self.dists[gtId][t].items(), key=lambda x: x[1], reverse=False)
        for item in sortedDists:
            if self.constellation.satDict[item[0]].orbitNum == orbitNum:
                return item[0]
        return None

    # closest active: init->closest, handover->when the closest changes, to->closest
    def closestActive(self):
        attachments = {}
        for gtId in tqdm(self.gtDict):
            gtAttachments = {}
            for t in range(0, self.constellation.SIM_PERIOD, STEP):
                gtAttachments[t] = self.getClosest(gtId, t)
            attachments[gtId] = gtAttachments
        return attachments

    # closest lazy: init->closest, handover->when the current satellite becomes invisible, to->closest
    def closestLazy(self):
        attachments = {}
        for gtId in tqdm(self.gtDict):
            gtAttachments = {}
            lastSat = None
            for t in range(0, self.constellation.SIM_PERIOD, STEP):
                tSatId = None
                if lastSat != None:
                    if lastSat in self.dists[gtId][t]:
                        tSatId = lastSat
                if tSatId == None: # covers two cases: 1.last sat invisible, 2.first time
                    tSatId = self.getClosest(gtId, t)
                gtAttachments[t] = tSatId
                lastSat = tSatId
            attachments[gtId] = gtAttachments
        return attachments

    # orbit closest active: init->closest, handover->orbit closest changes, or same orbit all invisible, to->closest in same orbit if any is visible, else closest
    def orbitClosestActive(self):
        attachments = {}
        for gtId in tqdm(self.gtDict):
            gtAttachments = {}
            lastSat = None
            for t in range(0, self.constellation.SIM_PERIOD, STEP):
                tSatId = None
                if lastSat != None:
                    tSatId = self.getOrbitClosest(lastSat, gtId, t)
                if tSatId == None:
                    tSatId = self.getClosest(gtId, t)
                gtAttachments[t] = tSatId
                lastSat = tSatId
            attachments[gtId] = gtAttachments
        return attachments

    # orbit closest lazy: init->closest, handover->invisible, to->closest in same orbit, else closest
    def orbitClosestLazy(self):
        attachments = {}
        for gtId in tqdm(self.gtDict):
            gtAttachments = {}
            lastSat = None
            for t in range(0, self.constellation.SIM_PERIOD, STEP):
                tSatId = None
                if lastSat != None:
                    if lastSat in self.dists[gtId][t]:
                        tSatId = lastSat
                    else:
                        tSatId = self.getOrbitClosest(lastSat, gtId, t)
                if tSatId == None:
                    tSatId = self.getClosest(gtId, t)
                gtAttachments[t] = tSatId
                lastSat = tSatId
            attachments[gtId] = gtAttachments
        return attachments

    # orbit closest active T: 
    #   init->choose an orbit that stays visible for the longest time, and pick the closest sat in that orbit
    #   handover->when the closest in the orbit changes
    #   to->the closest in the orbit if still visible, or go back to init and pick another orbit
    def orbitClosestActiveT(self):
        attachments = {}
        for gtId in tqdm(self.gtDict):
            gtAttachments = {}
            visibleOrbits = self.visibleOrbits[gtId]
            epochIndex = 0
            for t in range(0, self.constellation.SIM_PERIOD, STEP):
                tSatId = None
                if t >= visibleOrbits[epochIndex][0]:
                    epochIndex += 1
                curOrbit = visibleOrbits[epochIndex][1][0]
                tSatId = self.getOrbitClosestT(curOrbit, gtId, t)
                if tSatId == None:
                    print('fail')
                gtAttachments[t] = tSatId
            attachments[gtId] = gtAttachments
        return attachments

    # orbit closest lazy T: 
    #   init->choose an orbit that stays visible for the longest time, and pick the closest sat in that orbit
    #   handover->when the current becomes invisible
    #   to->the closest in the orbit if still visible, or go back to init and pick another orbit
    def orbitClosestLazyT(self):
        attachments = {}
        for gtId in tqdm(self.gtDict):
            gtAttachments = {}
            visibleOrbits = self.visibleOrbits[gtId]
            epochIndex = 0
            lastSat = None
            for t in range(0, self.constellation.SIM_PERIOD, STEP):
                tSatId = None
                if t >= visibleOrbits[epochIndex][0]:
                    epochIndex += 1
                curOrbit = visibleOrbits[epochIndex][1][0]
                if lastSat in self.dists[gtId][t]:
                        tSatId = lastSat
                else:
                        tSatId = self.getOrbitClosestT(curOrbit, gtId, t)
                if tSatId == None:
                    print('fail')
                gtAttachments[t] = tSatId
                lastSat = tSatId
            attachments[gtId] = gtAttachments
        return attachments


### utilities for analyzing forwarding paths

def parallelRun(func, argsList, chunksize=50):
    results = []
    with Pool() as p:
        with tqdm(total=len(argsList)) as pbar:
            for i, res in enumerate(p.imap_unordered(func, argsList, chunksize=chunksize)):
                results.append(res)
                pbar.update()
    return results

def getHandoverEpochs(attachments):
    epochsDict = {}
    for gtId in attachments:
        epochs = []
        atts = attachments[gtId]
        latt = None
        for t in atts:
            att = atts[t]
            if latt != None and att != latt:
                epochs.append(t)
            latt = att
        epochsDict[gtId] = epochs
    return epochsDict

def getProducerHandoverEpochs(scenario, gtId):
    epochs = []
    lastSats = None
    for t in scenario.constellation.snapshots:
        curSats = scenario.getVisibleSats(gtId, t)
        if (curSats != lastSats) and (lastSats != None):
            epochs.append(t)
        lastSats = curSats
    return epochs

class RouteInfo:
    def __init__(self, route, weight):
        self.route = route
        self.weight = weight

def _getPairRoutes(args): # for each consumer and producer handover: route after
    G, src, srcAtt, dest, destAtts, t = args # srcAtt is att after handover
    for att in destAtts:
        G.add_edge(dest, att, weight=99999)
    # routes = [r for r in nx.all_shortest_paths(G, source=src, target=dest, weight='weight')]
    # routeAfter = nx.shortest_path(G, source=srcAtt, target=dest, weight='weight')[:-1]
    routeAfter = [src] + nx.shortest_path(G, source=srcAtt, target=dest, weight='weight')
    weightAfter = path_weight(G, routeAfter[1:-1], weight="weight")
    return ((src, dest), t, RouteInfo(routeAfter, weightAfter))

def getPairRoutes(scenario, attStrategy, producerLinkAll=False): # c->p: routes after each consumer and producer handover
    attachments = scenario.getAttachments(attStrategy)
    epochsDict = getHandoverEpochs(attachments)
    gtPairs = list(permutations(scenario.gtDict.keys(), 2))
    argsList = []
    for gtPair in gtPairs:
        c = gtPair[0]
        p = gtPair[1]
        epochs = set(epochsDict[c]) | set(getProducerHandoverEpochs(scenario, p)) | set([0])
        for t in epochs:
            pAtts = None
            if producerLinkAll:
                pAtts = scenario.getVisibleSats(p, t)
            else:
                pAtts = [attachments[p][t]]
            argsList.append((scenario.constellation.snapshots[t], c, attachments[c][t], p, pAtts, t))
    raw = parallelRun(_getPairRoutes, argsList, chunksize=2000)
    routesDict = {}
    for item in raw:
        gtPair, t, routeInfo = item
        if not gtPair in routesDict:
            routesDict[gtPair] = {}
        routesDict[gtPair][t] = routeInfo
    for gtPair in routesDict:
        routes = routesDict[gtPair]
        epochs = sorted(routes.keys())
        lastRoute = routes[0]
        for t in epochs:
            if t == 0:
                continue
            curRoute = routes[t]
            if curRoute == lastRoute:
                del routes[t]
            lastRoute = curRoute
    return routesDict

def getRouteStats(pairRoutes): # pairRoutes: output of getHandverPairRoutes
    data = {
        'Epoch': [],
        'Consumer': [],
        'Producer': [],
        'PathLen': [],
        'LinkDistance': []
    }
    
    for gtPair in tqdm(pairRoutes):
        c = gtPair[0]
        p = gtPair[1]
        for t in pairRoutes[gtPair]:
            routeAfter = pairRoutes[gtPair][t].route
            data['Epoch'].append(t)
            data['Consumer'].append(c)
            data['Producer'].append(p)
            data['PathLen'].append(len(routeAfter)-1)
            data['LinkDistance'].append(pairRoutes[gtPair][t].weight)
    return pd.DataFrame.from_dict(data)

def _getHandoverPairRoutes(args): # for each consumer handover: route before, route after, and route between the previous and current access satellites
    G, src, srcAtt, dest, destAtts, t = args # srcAtt is (att before handover, att after handover)
    for att in destAtts:
        G.add_edge(dest, att, weight=99999)
    # routes = [r for r in nx.all_shortest_paths(G, source=src, target=dest, weight='weight')]
    routeBefore = nx.shortest_path(G, source=srcAtt[0], target=dest, weight='weight')[:-1]
    # weightBefore = path_weight(G, routeBefore, weight="weight")
    routeAfter = nx.shortest_path(G, source=srcAtt[1], target=dest, weight='weight')[:-1]
    # weightAfter = path_weight(G, routeAfter, weight="weight")
    routeBetween = nx.shortest_path(G, source=srcAtt[1], target=srcAtt[0], weight='weight') # from current to previous
    # weightBetween = path_weight(G, routeBetween, weight="weight")
    return (
            (src, dest), t,
            (RouteInfo(routeBefore, 0), RouteInfo(routeAfter, 0), RouteInfo(routeBetween, 0))
           )

def getHandoverPairRoutes(scenario, attStrategy, producerLinkAll=False): # c->p routes before and after each consumer handover, and routes between the previous and current access satellite
    attachments = scenario.getAttachments(attStrategy)
    epochsDict = getHandoverEpochs(attachments)
    gtPairs = list(permutations(scenario.gtDict.keys(), 2))
    argsList = []
    for gtPair in gtPairs:
        c = gtPair[0]
        p = gtPair[1]
        lastEpoch = 0
        for t in sorted(epochsDict[c]):
            pAtts = None
            if producerLinkAll:
                pAtts = scenario.getVisibleSats(p, t)
            else:
                pAtts = [attachments[p][t]]
            argsList.append((scenario.constellation.snapshots[t], c, (attachments[c][lastEpoch], attachments[c][t]), p, pAtts, t))
            lastEpoch = t
    raw = parallelRun(_getHandoverPairRoutes, argsList, chunksize=2000)
    routesDict = {}
    for item in raw:
        gtPair, t, routeInfos = item
        if not gtPair in routesDict:
            routesDict[gtPair] = {}
        routesDict[gtPair][t] = routeInfos
    return routesDict

def getCrossPoint(route1, route2): # return the cross point, and the hops from this cross point to the begin of each route
    done = False
    x = None
    for i in range(len(route1)):
        for j in range(len(route2)):
            if route2[j] == route1[i]:
                x = route1[i]
                done = True
                break
        if done:
            break
    return (x, i, j)

def getHandoverRouteStats(pairRoutes): # pairRoutes: output of getHandverPairRoutes
    data = {
        'Epoch': [],
        'Consumer': [],
        'Producer': [],
        'HitHops': [],
        'PathLenBefore': [],
        'PathLenAfter': [],
        'HandoverHops': [], # hops between the previous and current access satellite
        'HitHopsD': [] # hops between the previous and current access satellite
    }
    
    for gtPair in tqdm(pairRoutes):
        c = gtPair[0]
        p = gtPair[1]
        for t in pairRoutes[gtPair]:
            routeBefore = pairRoutes[gtPair][t][0].route
            routeAfter = pairRoutes[gtPair][t][1].route
            routeBetween = pairRoutes[gtPair][t][2].route
            # compute hit hops
            crossBA = getCrossPoint(routeAfter, routeBefore)
            crossBB = getCrossPoint(routeBetween, routeBefore)
            data['Epoch'].append(t)
            data['Consumer'].append(c)
            data['Producer'].append(p)
            data['HitHops'].append(crossBA[1])
            data['PathLenBefore'].append(len(routeBefore)-1)
            data['PathLenAfter'].append(len(routeAfter)-1)
            data['HandoverHops'].append(len(routeBetween))
            data['HitHopsD'].append(crossBB[1])
    return pd.DataFrame.from_dict(data)


### CZML section

from czml import czml

from datetime import datetime, timedelta, timezone

CZML_DIR = 'czml_files/'


### functions for adding entities

def genInterval(start, last, this):
    return '/'.join([(start+timedelta(seconds=last)).isoformat(), (start+timedelta(seconds=this)).isoformat()])

def genPolyline(id1, id2, intervals, color, suffix=''):
    polyId = 'line-%s-%s%s'%(id1, id2, suffix)
    packet = czml.CZMLPacket(id=polyId)
    sc = czml.SolidColor(color=color)
    m = czml.Material(solidColor=sc)
    refs = [id1+'#position', id2+'#position']
    pos = czml.Positions(references=refs)
    poly = czml.Polyline(width=7, followSurface=False, material=m, positions=pos)
    show = []
    avl = []
    for i in range(len(intervals)):
        show.append({'interval':intervals[i], 'show':True})
        avl.append(intervals[i])
    if len(intervals) == 0:
        show = False
    poly.show = show
    packet.polyline = poly
    packet.availability = avl
    return packet

# create and append the document packet
def initDoc(doc, start, end):
    packetDoc = czml.CZMLPacket(id='document',version='1.0')
    clock = czml.Clock()
    clock.interval = '/'.join([start.isoformat(), end.isoformat()])
    clock.currentTime = start.isoformat()
    clock.multiplier = 60
    clock.range = "LOOP_STOP",
    clock.step = "SYSTEM_CLOCK_MULTIPLIER"
    packetDoc.clock = clock
    doc.packets.append(packetDoc)

# add satellites
def genSatTrackCart(sat):
    global STEP
    track = []
    posList = sat.track.position.m
    for i in range(0, len(posList[0]), STEP):
        track.append(i)
        track.append(posList[0][i]) # x
        track.append(posList[1][i]) # y
        track.append(posList[2][i]) # z
    return track

def addSats(doc, satDict, start, end):
    print('Adding satellites to CZML file...')
    orbits = list(range(5))
    for satId in tqdm(satDict):
        # if not satDict[satId].orbitNum in orbits:
        #      continue
        packet = czml.CZMLPacket(id=satId)
        track = czml.Position()
        track.interpolationAlgorithm = 'LAGRANGE'
        track.interpolationDegree = 5
        track.referenceFrame = 'INERTIAL'
        track.epoch = start.isoformat()
        track.cartesian = genSatTrackCart(satDict[satId])
        packet.position = track
        sc = czml.SolidColor(color={'rgba': [0, 0, 0, 128]})
        m = czml.Material(solidColor=sc)
        if satDict[satId].satNum == 0:
            path = czml.Path()
            path.material = m
            path.width = 1.5
            path.show = True
            packet.path = path
        # bb = czml.Billboard(scale=1.5, show=True)
        # bb.image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAADJSURBVDhPnZHRDcMgEEMZjVEYpaNklIzSEfLfD4qNnXAJSFWfhO7w2Zc0Tf9QG2rXrEzSUeZLOGm47WoH95x3Hl3jEgilvDgsOQUTqsNl68ezEwn1vae6lceSEEYvvWNT/Rxc4CXQNGadho1NXoJ+9iaqc2xi2xbt23PJCDIB6TQjOC6Bho/sDy3fBQT8PrVhibU7yBFcEPaRxOoeTwbwByCOYf9VGp1BYI1BA+EeHhmfzKbBoJEQwn1yzUZtyspIQUha85MpkNIXB7GizqDEECsAAAAASUVORK5CYII="
        # bb.color = {'rgba': [255, 255, 255, 255]}
        # packet.billboard = bb
        point = czml.Point(show=True, color={'rgba': [0, 0, 0, 255]}, pixelSize=5)
        packet.point = point
        packet.availability = '/'.join([start.isoformat(), end.isoformat()])
        doc.packets.append(packet)

# add ISLs
def addISLs(doc, snapshots, start):
    print('Adding ISLs to CZML file...')
    ups = {}
    for t in snapshots:
        snapshot = snapshots[t]
        links = [genKey(link) for link in list(snapshot.edges)]
        for link in links:
            if not link in ups:
                ups[link] = {key: False for key in snapshots.keys()}
            ups[link][t] = True
    intervalDict = {}
    for link in ups:
        intervals = []
        epochs = sorted(ups[link].keys())
        t0 = None
        t1 = None
        for i in epochs:
            if ups[link][i]:
                if t0 == None:
                    t0 = i
                t1 = i+1
            else:
                if not t0 == None:
                    intervals.append(genInterval(start, t0, t1))
                    t0 = None
        if not t0 == None:
            intervals.append(genInterval(start, t0, len(epochs)))
        intervalDict[link] = intervals
    for link in tqdm(intervalDict):
        intervals = intervalDict[link]
        link = list(link)
        doc.packets.append(genPolyline(link[0], link[1], intervals, {'rgba': [0, 205, 0, 255]}))

# add GTs
def addGTs(doc, gtDict, start, end):
    print('Adding GTs to CZML file...')
    for gtId in tqdm(gtDict):
        packet = czml.CZMLPacket(id=gtId)
        pos = czml.Position()
        pos.cartographicRadians = [gtDict[gtId].longitude.degrees/180.*ephem.pi, gtDict[gtId].latitude.degrees/180.*ephem.pi, 0]
        packet.position = pos
        bb = czml.Billboard(scale=2, show=True)
        bb.image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAACvSURBVDhPrZDRDcMgDAU9GqN0lIzijw6SUbJJygUeNQgSqepJTyHG91LVVpwDdfxM3T9TSl1EXZvDwii471fivK73cBFFQNTT/d2KoGpfGOpSIkhUpgUMxq9DFEsWv4IXhlyCnhBFnZcFEEuYqbiUlNwWgMTdrZ3JbQFoEVG53rd8ztG9aPJMnBUQf/VFraBJeWnLS0RfjbKyLJA8FkT5seDYS1Qwyv8t0B/5C2ZmH2/eTGNNBgMmAAAAAElFTkSuQmCC"
        bb.color = {'rgba': [255, 255, 255, 255]}
        packet.billboard = bb
        packet.availability = '/'.join([start.isoformat(), end.isoformat()])
        doc.packets.append(packet)

# add user links
def addUserLinks(doc, satDict, gtDict, attachments, start, period):
    print('Computing link show intervals...')
    intervals = {}
    for gtId in tqdm(gtDict):
        lastSat = None
        lastTime = None
        intervals[gtId] = {}
        for satId in satDict:
            intervals[gtId][satId] = []
        for i in attachments[gtId]:
            if attachments[gtId][i] == None:
                if lastSat != None:
                    intervals[gtId][lastSat].append(genInterval(start, lastTime, i))
                    lastSat = None
                    lastTime = None
                else:
                    continue
            elif attachments[gtId][i] == lastSat:
                continue
            else:
                if lastSat != None:
                    intervals[gtId][lastSat].append(genInterval(start, lastTime, i))
                lastSat = attachments[gtId][i]
                lastTime = i
        if lastSat != None:
            intervals[gtId][lastSat].append(genInterval(start, lastTime, period-1))

    print('Adding user links to CZML file...')
    for gtId in tqdm(gtDict):
        for satId in satDict:
            doc.packets.append(genPolyline(gtId, satId, intervals[gtId][satId], {'rgba': [0, 0, 255, 255]}))

# add routes between two GTs
def addPairRoutes(doc, routes, start, period):
    print('Adding routes between a GT pair...')
    curLinkShow = {}
    lastLinkShow = {}
    lastPath = None
    for epoch in tqdm(sorted(routes.keys())):
        path = routes[epoch].route
        pathLinks = set()
        for i in range(1, len(path)):
            link = (path[i-1], path[i])
            pathLinks.add(link)
            if link not in curLinkShow:
                curLinkShow[link] = {key: False for key in routes.keys()}
            curLinkShow[link][epoch] = True
        if lastPath != None:
            lastPathLinks = set()
            for i in range(1, len(lastPath)):
                link = (lastPath[i-1], lastPath[i])
                lastPathLinks.add(link)
            for link in lastPathLinks-pathLinks:
                if link not in lastLinkShow:
                    lastLinkShow[link] = {key: False for key in routes.keys()}
                lastLinkShow[link][epoch] = True
        lastPath = path

    curRouteIntervals = {}
    for link in tqdm(curLinkShow):
        lastState = False
        lastTime = None
        curRouteIntervals[link] = []
        epochList = sorted(curLinkShow[link].keys())
        for i in epochList:
            if curLinkShow[link][i] == lastState:
                continue
            elif curLinkShow[link][i] == False:
                curRouteIntervals[link].append(genInterval(start, lastTime, i))
                lastState = False
                lastTime = None
            else:
                lastState = curLinkShow[link][i]
                lastTime = i
        if lastState == True:
            curRouteIntervals[link].append(genInterval(start, lastTime, period-1))
        curRouteIntervals[link].sort()
        
    for pair in tqdm(curRouteIntervals):
        doc.packets.append(genPolyline(pair[0], pair[1], curRouteIntervals[pair], {'rgba': [0, 255, 0, 255]}))

    lastRouteIntervals = {}
    for link in tqdm(lastLinkShow):
        lastState = False
        lastTime = None
        lastRouteIntervals[link] = []
        epochList = sorted(lastLinkShow[link].keys())
        for i in epochList:
            if lastLinkShow[link][i] == lastState:
                continue
            elif lastLinkShow[link][i] == False:
                lastRouteIntervals[link].append(genInterval(start, lastTime, i))
                lastState = False
                lastTime = None
            else:
                lastState = lastLinkShow[link][i]
                lastTime = i
        if lastState == True:
            lastRouteIntervals[link].append(genInterval(start, lastTime, period-1))
        lastRouteIntervals[link].sort()
        
    for pair in tqdm(lastRouteIntervals):
        doc.packets.append(genPolyline(pair[0], pair[1], lastRouteIntervals[pair], {'rgba': [255, 0, 0, 255]}, '-last'))

# generate producer routes
# def addGlobalRoutes(doc, routes, start):
#     intervals = {}
#     for epoch in tqdm(routes):
#         interval = genInterval(start, epoch, epoch+1)
#         route = routes[epoch]
#         for link in route:
#             if not link in intervals:
#                 intervals[link] = []
#             intervals[link].append(interval)
#     for link in tqdm(intervals):
#         doc.packets.append(genPolyline(link[0], link[1], intervals[link]))

# generate CZML for a constellation
def genConsCZML(constellation, filename):
    global CZML_DIR

    doc = czml.CZML()

    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=constellation.SIM_PERIOD)

    initDoc(doc, start, end)
    # addISLs(doc, constellation.snapshots, start)
    addSats(doc, constellation.satDict, start, end)

    print('Writing to CZML file...')
    doc.write(CZML_DIR+filename)
    print('Done!')

# generate CZML, in each GT pair, the first element is the consumer
def genCZML(scenario, attachments, routes, filename, gtPairs=None):
    global CZML_DIR

    doc = czml.CZML()

    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=scenario.constellation.SIM_PERIOD)
    
    gtDict = {}
    if gtPairs != None:
        for gtPair in gtPairs:
            gtDict[gtPair[0]] = None
            gtDict[gtPair[1]] = None
    for gtId in gtDict:
        gtDict[gtId] = scenario.gtDict[gtId]

    initDoc(doc, start, end)
    addSats(doc, scenario.constellation.satDict, start, end)
    addGTs(doc, gtDict, start, end)
    # addUserLinks(doc, scenario.constellation.satDict, gtDict, attachments, start, scenario.constellation.SIM_PERIOD)

    if gtPairs != None:
        for gtPair in gtPairs:
            addPairRoutes(doc, routes[gtPair], start, scenario.constellation.SIM_PERIOD)

    print('Writing to CZML file...')
    doc.write(CZML_DIR+filename)
    print('Done!')


### ndnSIM section, save attachments and routes for ndnSIM

NDNSIM_DIR = 'ndnsim_files/'

def storeNodes(scenario, dir):
    print('Storing nodes...')
    content = ['Name,Type\n']
    for satName in scenario.constellation.satDict:
        line = ','.join([satName, 'Satellite'])
        line += '\n'
        content.append(line)
    for gtName in scenario.gtDict:
        line = ','.join([gtName, 'Station'])
        line += '\n'
        content.append(line)
    fp = open(os.path.join(dir, 'nodes.csv'), 'w')
    fp.writelines(content)
    fp.close()

def storeISLs(scenario, dir):
    print('Storing ISLs...')
    header = ['First', 'Second']
    for t in scenario.constellation.snapshots:
        header.append(str(t))
    content = [','.join(header)+'\n']
    for edge in scenario.constellation.snapshots[0].edges: # now ISLs are all persistent
        line = ','.join([edge[0], edge[1]])
        for t in scenario.constellation.snapshots:
            delay = scenario.constellation.snapshots[t].get_edge_data(*edge)['weight']/300
            line = ','.join([line, str(delay)])
        line += '\n'
        content.append(line)
    fp = open(os.path.join(dir, 'ISLs.csv'), 'w')
    fp.writelines(content)
    fp.close()

def storeAttachments(scenario, dir):
    print('Storing attachments...')
    attachments = scenario.getAttachments('orbit closest active T')
    for gtId in attachments:
        content = ['Time,Satellite\n']
        lastSat = None
        for epoch in attachments[gtId]:
            line = None
            satId = attachments[gtId][epoch]
            if satId == None:
                satId = '-'
            if satId == lastSat:
                continue
            else:
                line = ','.join([str(epoch), satId])
                line += '\n'
            content.append(line)
            lastSat = satId
        fp = open(os.path.join(dir, 'attachments_%s.csv'%gtId), 'w')
        fp.writelines(content)
        fp.close()

# store GT pairs (consumer, producer) and the routes between each pair
def storeGtPairs(scenario, dir, gtPairs):
    gtPairContent = ['Consumer,Producer\n']
    routes = getHandoverPairRoutes(scenario, 'orbit closest active T', True)
    for gtPair in gtPairs:
        if gtPair[0] == gtPair[1]:
            continue
        print('Storing pair: %s -> %s...'%gtPair)
        content = ['Time,Route\n']
        route = routes[gtPair]
        # sort epochs
        epochs = sorted(list(route.keys()))
        for epoch in epochs:
            line = '|'.join(route[epoch][0].route)
            line = ','.join([str(epoch), line])
            line += '\n'
            content.append(line)
            line = '|'.join(route[epoch][1].route)
            line = ','.join([str(epoch), line])
            line += '\n'
            content.append(line)
        fp = open(os.path.join(dir, 'routes_%s+%s.csv'%(gtPair[0], gtPair[1])), 'w')
        fp.writelines(content)
        fp.close()
        gtPairContent.append(','.join(gtPair)+'\n')
    fp = open(os.path.join(dir, 'pairs.csv'), 'w')
    fp.writelines(gtPairContent)
    fp.close()

def genNdnSIM(scenario, gtPairs):
    global NDNSIM_DIR
    storeNodes(scenario, NDNSIM_DIR)
    storeISLs(scenario, NDNSIM_DIR)
    storeAttachments(scenario, NDNSIM_DIR)
    storeGtPairs(scenario, NDNSIM_DIR, gtPairs)
