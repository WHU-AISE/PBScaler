package plan.entity;

import lombok.Data;

import java.util.Date;
import java.util.List;

/**
 * @author fdse
 */
@Data
public class RoutePlanResultUnit {

    private String tripId;

    private String trainTypeId;

    private String fromStationName;

    private String toStationName;

    private List<String> stopStations;

    private String priceForSecondClassSeat;

    private String priceForFirstClassSeat;

    private Date startingTime;

    private Date endTime;

    public RoutePlanResultUnit() {
        //Default Constructor
    }

}
