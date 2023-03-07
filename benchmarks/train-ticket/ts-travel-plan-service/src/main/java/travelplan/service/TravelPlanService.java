package travelplan.service;

import edu.fudan.common.util.Response;
import org.springframework.http.HttpHeaders;
import travelplan.entity.*;

/**
 * @author fdse
 */
public interface TravelPlanService {

    Response getTransferSearch(TransferTravelInfo info, HttpHeaders headers);

    Response getCheapest(TripInfo info, HttpHeaders headers);

    Response getQuickest(TripInfo info, HttpHeaders headers);

    Response getMinStation(TripInfo info, HttpHeaders headers);

}
