package travel.entity;

import lombok.Data;

import java.util.List;

/**
 * @author fdse
 */
@Data
public class Route {

    private String id;

    private List<String> stations;

    private List<Integer> distances;

    private String startStationId;

    private String terminalStationId;

    public Route() {
        //Default Constructor
    }

    @Override
    public String toString() {
        return "Route{" +
                "id='" + id + '\'' +
                ", stations=" + stations +
                ", distances=" + distances +
                ", startStationId='" + startStationId + '\'' +
                ", terminalStationId='" + terminalStationId + '\'' +
                '}';
    }
}
