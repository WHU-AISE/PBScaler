package admintravel.entity;

import java.util.Date;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author fdse
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class TravelInfo {
    private String loginId;

    private String tripId;

    private String trainTypeId;

    private String routeId;

    private String startingStationId;

    private String stationsId;

    private String terminalStationId;

    private Date startingTime;

    private Date endTime;

}
