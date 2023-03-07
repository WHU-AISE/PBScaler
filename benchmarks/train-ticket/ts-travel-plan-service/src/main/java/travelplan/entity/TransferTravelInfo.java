package travelplan.entity;

import lombok.Data;

import java.util.Date;

/**
 * @author fdse
 */
@Data
public class TransferTravelInfo {

    private String fromStationName;

    private String viaStationName;

    private String toStationName;

    private Date travelDate;

    private String trainType;

    public TransferTravelInfo() {
        //Empty Constructor
    }

    public TransferTravelInfo(String fromStationName, String viaStationName, String toStationName, Date travelDate, String trainType) {
        this.fromStationName = fromStationName;
        this.viaStationName = viaStationName;
        this.toStationName = toStationName;
        this.travelDate = travelDate;
        this.trainType = trainType;
    }

}
