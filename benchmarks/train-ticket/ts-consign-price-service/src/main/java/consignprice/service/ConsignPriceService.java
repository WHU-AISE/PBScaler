package consignprice.service;

import consignprice.entity.ConsignPrice;
import edu.fudan.common.util.Response;
import org.springframework.http.HttpHeaders;

/**
 * @author fdse
 */
public interface ConsignPriceService {

    /**
     * get price by weight and region
     *
     * @param weight weight
     * @param isWithinRegion whether is within region
     * @param headers headers
     * @return Response
     */
    Response getPriceByWeightAndRegion(double weight, boolean isWithinRegion, HttpHeaders headers);

    /**
     * query price information
     *
     * @param headers headers
     * @return Response
     */
    Response queryPriceInformation(HttpHeaders headers);

    /**
     * create and modify price
     *
     * @param config config
     * @param headers headers
     * @return Response
     */
    Response createAndModifyPrice(ConsignPrice config, HttpHeaders headers);

    /**
     * get price config
     *
     * @param headers headers
     * @return Response
     */
    Response getPriceConfig(HttpHeaders headers);
}
