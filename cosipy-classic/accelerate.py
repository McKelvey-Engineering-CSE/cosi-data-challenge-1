from numba import jit, njit, prange
import numpy as np
import ROOT as M
from numba.typed import List
import numba

@jit(fastmath=True, cache=True, nogil=True)
def accel_read_COSI_DataSet(Reader, erg, tt, et,
                            latX, lonX, latZ, lonZ, phi,
                            chi_loc, psi_loc, dist,
                            chi_gal, psi_gal, n_events) :
    
    # browse through .tra file, select events, and sort into corresponding list
    n_event = 0
    while True:
        # the Reader class from MEGAlib knows where an event starts and ends and
        # returns the Event object which includes all information of an event
        Event = Reader.GetNextEvent()
        if not Event:
            break

        #if n_event%10000 == 0 :
        #    print("Reading event", n_event)

        if n_events > 0 and n_event == n_events :
             return
        n_event = n_event+1
        
        # here only select Compton events (will add Photo events later as optional)

        # all calculations and definitions taken from:
        # /MEGAlib/src/response/src/MResponseImagingBinnedMode.cxx
            
        # Total Energy
        erg.append(Event.Ei())
        # Time tag in UNIX seconds
        tt.append(Event.GetTime().GetAsSeconds())
        # Event type (0 = Compton, 4 = Photo)
        et.append(Event.GetEventType())
        # x axis of space craft pointing at GAL latitude
        latX.append(Event.GetGalacticPointingXAxisLatitude())
        # x axis of space craft pointing at GAL longitude
        lonX.append(Event.GetGalacticPointingXAxisLongitude())
        # z axis of space craft pointing at GAL latitude
        latZ.append(Event.GetGalacticPointingZAxisLatitude())
        # z axis of space craft pointing at GAL longitude
        lonZ.append(Event.GetGalacticPointingZAxisLongitude()) 
        
        # note that the y axis can be calculated from the X and Z components
        # therefore it is not saved, and will be computed further down
            
        if Event.GetEventType() == M.MPhysicalEvent.c_Compton:    
            # Compton scattering angle
            phi.append(Event.Phi()) 
            # data space angle chi (azimuth)
            chi_loc.append((-Event.Dg()).Phi())
            # data space angle psi (polar)
            psi_loc.append((-Event.Dg()).Theta())
            # interaction length between first and second scatter in cm
            dist.append(Event.FirstLeverArm())
            # gal longitude angle corresponding to chi
            chi_gal.append((Event.GetGalacticPointingRotationMatrix()*Event.Dg()).Phi())
            # gal longitude angle corresponding to chi
            psi_gal.append((Event.GetGalacticPointingRotationMatrix()*Event.Dg()).Theta())



@njit(fastmath=True, parallel=True, nogil=True)
def accel_get_binned_data(binned_data, n_ph, n_ph_dx,
                          n_energy_bins, n_fisbel_bins,
                          energy_bin_edges, phi_edges,
                          lon_min, lon_max, lat_min, lat_max,
                          data_time_tagged_indices,
                          phi, psi, chi, erg) :
    

    # Loop over time bins with events
    for ph_dx in prange(n_ph) :

        
        # get the event indices within this time bin
        idx_tmp = data_time_tagged_indices[n_ph_dx[ph_dx]]

        # get the corresponding angles, energies,
        # and lat/longs of those events
        phi_tmp = phi[idx_tmp]
        psi_tmp = psi[idx_tmp]
        chi_tmp = chi[idx_tmp]
        erg_tmp = erg[idx_tmp]
        
        # loop over FISBEL bins
        for f in range(n_fisbel_bins) :

            # Get the range of lats and longs in that FISBEL, i.e.,
            # those indices as a subset of idx_tmp where
            # chi and psi are within the FISBEL
            fisbel_idx_tmp = np.where((chi_tmp >= lon_min[f]) &
                                    (chi_tmp < lon_max[f]) &
                                    (psi_tmp >= lat_min[f]) &
                                    (psi_tmp < lat_max[f]))[0]  

            phi_tmp_fisbel = phi_tmp[fisbel_idx_tmp]          
            erg_tmp_fisbel = erg_tmp[fisbel_idx_tmp]     

            # Since numba doesn't natively support 2D histograms,
            # We loop over energy bins,
            # constructing a 1D histogram over phi_edges for each energy

            # All but last bin are half open
            for e in range(n_energy_bins - 1) :

                energy_idx_tmp = np.where((erg_tmp_fisbel >= energy_bin_edges[e]) &
                                        (erg_tmp_fisbel < energy_bin_edges[e+1]))[0]
                    
                
                hist_tmp = np.histogram(phi_tmp_fisbel[energy_idx_tmp], bins = phi_edges)
                
                binned_data[ph_dx, e, :, f] = hist_tmp[0]

            # Last bin is fully closed
            e = n_energy_bins - 1
            energy_idx_tmp = np.where((erg_tmp_fisbel >= energy_bin_edges[e]) &
                                    (erg_tmp_fisbel <= energy_bin_edges[e+1]))[0]
                
            
            binned_data[ph_dx, e, :, f] = np.histogram(phi_tmp_fisbel[energy_idx_tmp], bins = phi_edges)[0]

x = np.zeros(1).astype(int)
numba_np_type = numba.typeof(x)

@njit(fastmath=True, parallel=True, nogil=True)
def accel_time_binning_tags(n_time_bins, time_bin_size, last_bin_size,
                            TimeTags, data_delta_times,
                            times_edges, n_ph_t,
                            times_min, times_max,
                            times_cen, times_wid) :

    s2b = 1./time_bin_size

    # Our list of arrays of indices needs to be initialized with the correct type,
    # then filled with something trivial to allow access-by-index
    # instead of appends, which can't be parallelized
    data_time_indices = List.empty_list(numba_np_type)
    data_time_indices = [x] * n_time_bins
    
    # Shift times so smallest time is 0
    minmin_TimeTags = TimeTags - np.min(TimeTags)

    # Parallel loop over time bins
    for b in prange(n_time_bins):

        # Event indices in the time bin
        tdx = np.where( (minmin_TimeTags*s2b >= b) &
                        (minmin_TimeTags*s2b <  b+1) )[0]
        
        # Write indices and length
        data_time_indices[b] = tdx
        n_ph_t[b] = len(tdx)
        
        # Compute histogram stats
        # (edges, half widths, etc)
        edge = b*time_bin_size
        times_edges[b] = edge
        times_min[b] = edge
        times_max[b] = edge + time_bin_size
        times_cen[b] = edge + time_bin_size * 0.5
        times_wid[b] = time_bin_size * 0.5
        data_delta_times[b] = time_bin_size
        
    # Last histogram bin is sized to fit, so it may be shorter
    # Again, might not be necessary, but we can deal with this later
    last_edge = (n_time_bins - 1) * time_bin_size + last_bin_size
    times_edges[n_time_bins] = last_edge
    times_max[n_time_bins - 1] = last_edge
    times_cen[n_time_bins - 1] = last_edge - last_bin_size * 0.5
    times_wid[n_time_bins - 1] = last_bin_size * 0.5
    data_delta_times[n_time_bins - 1] = last_bin_size

    return data_time_indices
