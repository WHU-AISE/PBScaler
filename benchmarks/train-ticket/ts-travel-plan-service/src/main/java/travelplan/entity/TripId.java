package travelplan.entity;

import lombok.Data;

import java.io.Serializable;

/**
 * @author fdse
 */
@Data
public class TripId implements Serializable{

    private TrainTypeEnum type;

    private String number;

    public TripId(){
        //Default Constructor
    }

    public TripId(String trainNumber){
        char type0 = trainNumber.charAt(0);
        switch(type0){
            case 'G': this.type = TrainTypeEnum.G;
                break;
            case 'D': this.type = TrainTypeEnum.D;
                break;
            case 'Z': this.type = TrainTypeEnum.Z;
                break;
            case 'T': this.type = TrainTypeEnum.T;
                break;
            case 'K': this.type = TrainTypeEnum.K;
                break;
            default:break;
        }

        this.number = trainNumber.substring(1);
    }

    @Override
    public String toString(){
        return type.getName() + number;
    }
}
