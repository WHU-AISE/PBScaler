package route.entity;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * @author fdse
 */
@Data
@AllArgsConstructor
public class RouteInfo {
    private String id;

    private String startStation;

    private String endStation;

    private String stationList;

    private String distanceList;

    public RouteInfo() {
        //Default Constructor
    }

}
