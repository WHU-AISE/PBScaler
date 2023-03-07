package travel.entity;

import lombok.Data;

import java.util.Date;

/**
 * @author fdse
 */
@Data
public class TravelInfo {

    private String tripId;

    private String trainTypeId;

    private String routeId;

    private String startingStationId;

    private String stationsId;

    private String terminalStationId;

    private Date startingTime;

    private Date endTime;

    public TravelInfo() {
        //Default Constructor
    }

}
